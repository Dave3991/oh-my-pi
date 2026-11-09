import { afterAll, afterEach, beforeEach, describe, expect, it, spyOn, vi } from "bun:test";
import * as fs from "node:fs/promises";
import * as os from "node:os";
import * as path from "node:path";
import { type } from "@oh-my-pi/omptype";
import type { AgentTool, AgentToolContext, AgentToolResult, SpeculativeOperationSink } from "@oh-my-pi/pi-agent-core";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import * as jsContextManager from "@oh-my-pi/pi-coding-agent/eval/js/context-manager";
import { callSessionTool } from "@oh-my-pi/pi-coding-agent/eval/js/tool-bridge";
import type { EvalPreludeDefinition } from "@oh-my-pi/pi-coding-agent/eval/preludes";
import { disposeAllKernelSessions } from "@oh-my-pi/pi-coding-agent/eval/py/executor";
import {
	disposePyToolBridge,
	ensurePyToolBridge,
	registerPyToolBridge,
} from "@oh-my-pi/pi-coding-agent/eval/py/tool-bridge";
import { EvalShadowCellSession } from "@oh-my-pi/pi-coding-agent/eval/speculation/cell-session";
import { ArtifactProtocolHandler } from "@oh-my-pi/pi-coding-agent/internal-urls/artifact-protocol";
import { parseInternalUrl } from "@oh-my-pi/pi-coding-agent/internal-urls/parse";
import {
	artifactsDirsFromRegistry,
	resetRegisteredArtifactDirsForTests,
} from "@oh-my-pi/pi-coding-agent/internal-urls/registry-helpers";
import { AgentRegistry } from "@oh-my-pi/pi-coding-agent/registry/agent-registry";
import type { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import type { Tool, ToolSession } from "@oh-my-pi/pi-coding-agent/tools";
import { BashTool } from "@oh-my-pi/pi-coding-agent/tools/bash";
import { EvalTool } from "@oh-my-pi/pi-coding-agent/tools/eval";
import { GlobTool } from "@oh-my-pi/pi-coding-agent/tools/glob";
import { postProcessToolResult, wrapToolWithMetaNotice } from "@oh-my-pi/pi-coding-agent/tools/output-meta";
import { ReadTool } from "@oh-my-pi/pi-coding-agent/tools/read";
import { WriteTool } from "@oh-my-pi/pi-coding-agent/tools/write";
import { resolveXdevTool, type XdevState } from "@oh-my-pi/pi-coding-agent/tools/xdev";
import { $which, isRecord, removeWithRetries } from "@oh-my-pi/pi-utils";

/** A pretty-printed JSON document well past the 50 KB spill threshold. */
function bigJson(label: string): string {
	const items = Array.from({ length: 3_000 }, (_, index) => ({
		index,
		name: `${label}-item-${index}`,
		padding: "x".repeat(24),
	}));
	return JSON.stringify({ label, items }, null, 1);
}

function textOf(result: AgentToolResult): string {
	return result.content
		.filter((block): block is { type: "text"; text: string } => block.type === "text")
		.map(block => block.text)
		.join("");
}

interface BridgeObject {
	text: string;
	details?: unknown;
	hasError?: unknown;
	incomplete?: unknown;
}

function asObject(value: unknown): BridgeObject {
	if (!isRecord(value) || typeof value.text !== "string") {
		throw new Error(`expected a bridge object, got ${JSON.stringify(value)?.slice(0, 200)}`);
	}
	return { text: value.text, details: value.details, hasError: value.hasError, incomplete: value.incomplete };
}

/** `meta.truncation` recorded at one level of a result's details. */
function truncationOf(details: unknown): Record<string, unknown> | undefined {
	if (!isRecord(details) || !isRecord(details.meta) || !isRecord(details.meta.truncation)) return undefined;
	return details.meta.truncation;
}

/** The tool session a real SDK session hands the eval bridge: its own journal, artifact manager and tool context. */
function ownerSession(cwd: string, manager: SessionManager, overrides: Partial<ToolSession> = {}): ToolSession {
	const settings = Settings.isolated();
	const context = { sessionManager: manager, settings } as unknown as AgentToolContext;
	return {
		cwd,
		hasUI: false,
		getSessionFile: () => manager.getSessionFile() ?? null,
		getSessionSpawns: () => "*",
		settings,
		sessionManager: manager,
		getArtifactManager: () => manager.getArtifactManager(),
		getToolContext: () => context,
		...overrides,
	};
}

/** A tool whose own meta wrapper spills its output, as every registered session tool is wrapped. */
function wrappedTool(name: string, text: string, details?: Record<string, unknown>): AgentTool {
	return wrapToolWithMetaNotice({
		name,
		label: name,
		description: `${name} device`,
		parameters: type({}),
		approval: () => "read",
		async execute() {
			return { content: [{ type: "text", text }], ...(details ? { details } : {}) };
		},
	} as AgentTool);
}

/** A tool returning a prepared result verbatim (no wrapper). */
function fixedTool(name: string, result: AgentToolResult, parameters: unknown = type({})): AgentTool {
	return {
		name,
		label: name,
		description: name,
		parameters,
		async execute() {
			return result;
		},
	} as unknown as AgentTool;
}

function withTools(session: ToolSession, tools: AgentTool[]): ToolSession {
	const registry = new Map(tools.map(tool => [tool.name, tool]));
	session.getToolByName = name => registry.get(name);
	return session;
}

function createTestXdevState(tools: AgentTool[]): XdevState {
	const state: XdevState = {
		tools: new Map(tools.map(tool => [tool.name, tool as Tool])),
		mountedNames: new Set(tools.map(tool => tool.name)),
		builtInNames: new Set(tools.map(tool => tool.name)),
		isActive: () => false,
		resolve: name => resolveXdevTool(state, name),
	};
	return state;
}

/** Minimal `meta.truncation` record; `artifactId` names the full output, when one was saved. */
function cut(artifactId: string | undefined, extra: Record<string, unknown> = {}): Record<string, unknown> {
	return {
		direction: "middle",
		truncatedBy: "middle",
		totalLines: 1,
		totalBytes: 1,
		outputLines: 1,
		outputBytes: 1,
		...(artifactId === undefined ? {} : { artifactId }),
		...extra,
	};
}

/** Minimal `meta.limits.columnTruncated` record: lines cut to the column cap; `artifactId` holds them whole. */
function columnCut(artifactId: string | undefined, extra: Record<string, unknown> = {}): Record<string, unknown> {
	return { maxColumn: 768, unit: "bytes", ...(artifactId === undefined ? {} : { artifactId }), ...extra };
}

/** A real spill of `text` into `manager`: the cut result a wrapped tool returns. */
async function spilled(manager: SessionManager, name: string, text: string): Promise<AgentToolResult> {
	const context = { sessionManager: manager, settings: Settings.isolated() } as unknown as AgentToolContext;
	const result = await postProcessToolResult({ content: [{ type: "text", text }], details: {} }, name, context);
	if (!truncationOf(result.details)?.artifactId) throw new Error("expected the spill to save an artifact");
	return result;
}

describe("eval bridge returns complete tool-result text", () => {
	let tempDir: string;
	const managers: SessionManager[] = [];

	beforeEach(async () => {
		tempDir = await fs.mkdtemp(path.join(os.tmpdir(), "bridge-full-text-"));
	});

	afterEach(async () => {
		vi.restoreAllMocks();
		AgentRegistry.resetGlobalForTests();
		resetRegisteredArtifactDirsForTests();
		await Promise.all(managers.splice(0).map(manager => manager.close()));
		await removeWithRetries(tempDir);
	});

	afterAll(async () => {
		await jsContextManager.disposeAllVmContexts();
		await disposeAllKernelSessions();
		await disposePyToolBridge();
	});

	/** A persistent session: artifacts are files in its own artifacts dir. */
	function fileManager(): SessionManager {
		const manager = SessionManager.create(tempDir, path.join(tempDir, "sessions"));
		managers.push(manager);
		return manager;
	}

	/** A `--no-session` / SDK session: artifacts stay in memory. */
	function memoryManager(): SessionManager {
		const manager = SessionManager.inMemory(tempDir);
		managers.push(manager);
		return manager;
	}

	it("(a) restores the full JSON of a device written through xd:// whose own wrapper cut it", async () => {
		const payload = bigJson("device");
		const manager = fileManager();
		const xdev = createTestXdevState([wrappedTool("big_json", payload)]);
		const session = ownerSession(tempDir, manager, { xdev });
		// The registry erases each tool's parameter type, as the session's does.
		const write = wrapToolWithMetaNotice(new WriteTool(session)) as AgentTool;
		withTools(session, [write]);

		const value = asObject(await callSessionTool("write", { path: "xd://big_json", content: "{}" }, { session }));

		// The cut is recorded by the device under `xdev.inner`, not at the top level.
		expect(truncationOf(value.details)).toBeUndefined();
		const inner = isRecord(value.details) && isRecord(value.details.xdev) ? value.details.xdev.inner : undefined;
		expect(truncationOf(inner)?.artifactId).toBeDefined();
		expect(value.text).toBe(payload);
		expect(JSON.parse(value.text).items).toHaveLength(3_000);
		expect(value.incomplete).toBeUndefined();
	});

	it("(b) lets the deepest xd level's cut decide", async () => {
		const manager = fileManager();
		const full = bigJson("producer");
		const innerId = (await manager.saveArtifact(full, "inner"))!;
		const outerId = (await manager.saveArtifact("outer cut of the producer's preview", "outer"))!;
		const nest = (inner: Record<string, unknown>) => ({ xdev: { tool: "device", mode: "execute", inner } });
		const cases: Array<{ label: string; details: Record<string, unknown>; expected: string }> = [
			{
				label: "outer cut over inner cut",
				details: { meta: { truncation: cut(outerId) }, ...nest({ meta: { truncation: cut(innerId) } }) },
				expected: full,
			},
			{
				label: "inner cut alone, two xd levels deep",
				details: nest(nest({ meta: { truncation: cut(innerId) } })),
				expected: full,
			},
			{
				label: "only the outer level cut",
				details: { meta: { truncation: cut(outerId) }, ...nest({ meta: {} }) },
				expected: "outer cut of the producer's preview",
			},
		];
		for (const { label, details, expected } of cases) {
			const session = withTools(ownerSession(tempDir, manager), [
				fixedTool("nested", { content: [{ type: "text", text: "displayed preview" }], details }),
			]);
			const value = asObject(await callSessionTool("nested", {}, { session }));
			expect({ label, text: value.text, incomplete: value.incomplete }).toEqual({
				label,
				text: expected,
				incomplete: undefined,
			});
		}
	});

	it("(c) reports a sample artifact as incomplete and keeps the displayed text", async () => {
		const manager = fileManager();
		const id = (await manager.saveArtifact("head …sample… tail", "bash"))!;
		const session = withTools(ownerSession(tempDir, manager), [
			fixedTool("sampled", {
				content: [{ type: "text", text: "displayed preview" }],
				details: { meta: { truncation: cut(id, { artifactElidedBytes: 4_096 }) } },
			}),
		]);

		const value = asObject(await callSessionTool("sampled", {}, { session }));

		expect(value.text).toBe("displayed preview");
		expect(value.incomplete).toEqual({ artifactId: id, reason: "sample" });
	});

	it("(d) never fills a missing owning-session artifact from another session with the same id", async () => {
		const manager = fileManager();
		const session = withTools(ownerSession(tempDir, manager), [wrappedTool("big_json", bigJson("own"))]);
		const foreignDir = path.join(tempDir, "foreign-artifacts");
		await fs.mkdir(foreignDir);
		AgentRegistry.global().register({
			id: "Foreign",
			displayName: "Foreign",
			kind: "sub",
			status: "idle",
			session: { sessionManager: { getArtifactsDir: () => foreignDir } } as unknown as AgentSession,
		});
		// By the time the bridge reads it, the owner's artifact file is gone, while another live
		// session holds a file under the same per-session id.
		const realSave = manager.saveArtifact.bind(manager);
		let savedId: string | undefined;
		spyOn(manager, "saveArtifact").mockImplementation(async (content, toolType) => {
			savedId = await realSave(content, toolType);
			if (savedId !== undefined) {
				const ownPath = await manager.getArtifactPath(savedId);
				if (ownPath) await fs.rm(ownPath);
				await Bun.write(path.join(foreignDir, `${savedId}.big_json.log`), '{"FOREIGN": true}');
			}
			return savedId;
		});

		const value = asObject(await callSessionTool("big_json", {}, { session }));

		if (savedId === undefined) throw new Error("expected the call to spill");
		// The registry search would hand out the other session's bytes for this id.
		expect(artifactsDirsFromRegistry()).toContain(foreignDir);
		const foreign = await new ArtifactProtocolHandler().resolve(parseInternalUrl(`artifact://${savedId}`));
		expect(foreign.content).toContain("FOREIGN");
		// The bridge reads only the owning session and says the text is incomplete.
		expect(value.incomplete).toEqual({ artifactId: savedId, reason: "unreadable" });
		expect(value.text).not.toContain("FOREIGN");
		expect(value.text).toContain(`artifact://${savedId}`);
	});

	it("(e) restores an in-memory spill of a session without an artifacts dir", async () => {
		const payload = bigJson("memory");
		const manager = memoryManager();
		expect(manager.getArtifactsDir()).toBeNull();
		const session = withTools(ownerSession(tempDir, manager), [wrappedTool("big_json", payload)]);

		const value = asObject(await callSessionTool("big_json", {}, { session }));

		expect(value.text).toBe(payload);
		expect(value.incomplete).toBeUndefined();
	});

	it("(f) restores the full text of a directly called wrapped tool", async () => {
		const payload = bigJson("direct");
		const session = withTools(ownerSession(tempDir, fileManager()), [wrappedTool("big_json", payload)]);
		const statuses: Array<Record<string, unknown>> = [];

		const value = asObject(
			await callSessionTool("big_json", {}, { session, emitStatus: event => statuses.push(event) }),
		);

		expect(value.text).toBe(payload);
		expect(value.incomplete).toBeUndefined();
		// The status line still summarizes what was displayed, not the restored text.
		expect(statuses).toHaveLength(1);
		expect(statuses[0]?.op).toBe("big_json");
		expect(statuses[0]?.chars).toBeLessThan(payload.length);
	});

	it("(g) leaves small results unchanged", async () => {
		const session = withTools(ownerSession(tempDir, fileManager()), [
			wrappedTool("small", "small output"),
			wrappedTool("small_details", "ok", { count: 1 }),
		]);

		expect(await callSessionTool("small", {}, { session })).toBe("small output");
		expect(await callSessionTool("small_details", {}, { session })).toEqual({ text: "ok", details: { count: 1 } });
	});

	it("reports a cut whose spill could not be saved as unreadable, keeping the preview, without a replay", async () => {
		const payload = bigJson("unsaved");
		const manager = memoryManager();
		spyOn(manager, "saveArtifact").mockRejectedValue(new Error("disk full"));
		// Counts runs of the tool's own body, below its meta wrapper: the side effect a replay would repeat.
		let executions = 0;
		const tool = wrapToolWithMetaNotice({
			name: "big_json",
			label: "big_json",
			description: "big_json device",
			parameters: type({}),
			approval: () => "read",
			async execute() {
				executions += 1;
				return { content: [{ type: "text", text: payload }] };
			},
		} as AgentTool);
		const session = withTools(ownerSession(tempDir, manager), [tool]);

		const value = asObject(await callSessionTool("big_json", {}, { session }));

		const truncation = truncationOf(value.details);
		expect(truncation).toBeDefined();
		expect(truncation?.artifactId).toBeUndefined();
		expect(value.text.length).toBeLessThan(payload.length);
		expect(value.text).not.toContain("artifact://");
		expect(value.incomplete).toEqual({ reason: "unreadable" });
		expect(isRecord(value.incomplete) && Object.hasOwn(value.incomplete, "artifactId")).toBe(false);
		// Restoration failed, and the bridge did not run the tool again to get the text another way.
		expect(executions).toBe(1);
	});

	it("(i) restores a bash output line the sink cut to its column cap", async () => {
		const manager = fileManager();
		const items = Array.from({ length: 200 }, (_, index) => ({ index, name: `wide-item-${index}` }));
		const minified = JSON.stringify({ label: "wide", items });
		await Bun.write(path.join(tempDir, "doc.json"), minified);
		const session = ownerSession(tempDir, manager, {
			settings: Settings.isolated({
				"async.enabled": false,
				"bash.autoBackground.enabled": false,
				"bashInterceptor.enabled": false,
			}),
			skills: [],
			getClientBridge: () => undefined,
			allocateOutputArtifact: toolType => manager.allocateArtifactPath(toolType),
		});
		// The registry erases each tool's parameter type, as the session's does.
		const bash = wrapToolWithMetaNotice(new BashTool(session)) as AgentTool;
		withTools(session, [bash]);

		const value = asObject(await callSessionTool("bash", { command: "cat doc.json" }, { session }));

		// One line over the sink's default column cap, far under the window limit: only a column cut is recorded.
		expect(minified.length).toBeGreaterThan(768);
		expect(truncationOf(value.details)).toBeUndefined();
		const meta = isRecord(value.details) && isRecord(value.details.meta) ? value.details.meta : undefined;
		const column = isRecord(meta?.limits) ? meta.limits.columnTruncated : undefined;
		expect(isRecord(column) ? column.artifactId : undefined).toBeDefined();
		expect(value.text).toBe(minified);
		expect(JSON.parse(value.text).items).toHaveLength(200);
		expect(value.incomplete).toBeUndefined();
	});

	it("(ii) lets the deepest level's column cut decide over an outer window cut", async () => {
		const manager = fileManager();
		const full = bigJson("column-producer");
		const innerId = (await manager.saveArtifact(full, "inner"))!;
		const outerId = (await manager.saveArtifact("OUTER cut of the column-cut preview", "outer"))!;
		const underOuterCut = (inner: Record<string, unknown>) => ({
			meta: { truncation: cut(outerId) },
			xdev: { tool: "device", mode: "execute", inner },
		});
		const cases: Array<{ label: string; details: Record<string, unknown>; text: string; incomplete: unknown }> = [
			{
				label: "inner column cut with its artifact",
				details: underOuterCut({ meta: { limits: { columnTruncated: columnCut(innerId) } } }),
				text: full,
				incomplete: undefined,
			},
			{
				label: "inner column cut without an artifact",
				details: underOuterCut({ meta: { limits: { columnTruncated: columnCut(undefined) } } }),
				text: "displayed preview",
				incomplete: { reason: "unreadable" },
			},
		];
		for (const { label, details, text, incomplete } of cases) {
			const session = withTools(ownerSession(tempDir, manager), [
				fixedTool("nested", { content: [{ type: "text", text: "displayed preview" }], details }),
			]);
			const value = asObject(await callSessionTool("nested", {}, { session }));
			expect({ label, text: value.text, incomplete: value.incomplete }).toEqual({ label, text, incomplete });
		}
	});

	it("(iii) reports a column cut whose artifact is a sample as incomplete", async () => {
		const manager = fileManager();
		const id = (await manager.saveArtifact("head …sample… tail", "bash"))!;
		const session = withTools(ownerSession(tempDir, manager), [
			fixedTool("sampled", {
				content: [{ type: "text", text: "displayed preview" }],
				details: { meta: { limits: { columnTruncated: columnCut(id, { artifactElidedBytes: 4_096 }) } } },
			}),
		]);

		const value = asObject(await callSessionTool("sampled", {}, { session }));

		expect(value.text).toBe("displayed preview");
		expect(value.incomplete).toEqual({ artifactId: id, reason: "sample" });
	});

	it("(iv) restores a window cut and a column cut at one level from their shared artifact", async () => {
		const manager = fileManager();
		const full = bigJson("both-cuts");
		const id = (await manager.saveArtifact(full, "bash"))!;
		const otherId = (await manager.saveArtifact("another output", "bash"))!;
		const cases: Array<{ label: string; details: Record<string, unknown>; text: string; incomplete: unknown }> = [
			{
				label: "one shared artifact",
				details: { meta: { truncation: cut(id), limits: { columnTruncated: columnCut(id) } } },
				text: full,
				incomplete: undefined,
			},
			{
				label: "two different artifacts",
				details: { meta: { truncation: cut(id), limits: { columnTruncated: columnCut(otherId) } } },
				text: "displayed preview",
				incomplete: { reason: "unreadable" },
			},
		];
		for (const { label, details, text, incomplete } of cases) {
			const session = withTools(ownerSession(tempDir, manager), [
				fixedTool("both", { content: [{ type: "text", text: "displayed preview" }], details }),
			]);
			const value = asObject(await callSessionTool("both", {}, { session }));
			expect({ label, text: value.text, incomplete: value.incomplete }).toEqual({ label, text, incomplete });
		}
	});

	it("(v) reports glob's artifact-less cut as incomplete after the spill cuts it again", async () => {
		// One match per directory, both names long: the grouped listing passes glob's fixed 50 KB cap.
		const root = path.join(tempDir, "listing");
		for (let index = 0; index < 150; index++) {
			const suffix = String(index).padStart(3, "0");
			const dir = path.join(root, `${"d".repeat(200)}-${suffix}`);
			await fs.mkdir(dir, { recursive: true });
			await Bun.write(path.join(dir, `${"f".repeat(200)}-${suffix}.txt`), "");
		}
		const manager = fileManager();
		// A 20 KB spill threshold, so glob's 50 KB cut is cut once more.
		const settings = Settings.isolated({ "tools.artifactSpillThreshold": 20 });
		const context = { sessionManager: manager, settings } as unknown as AgentToolContext;
		const session = ownerSession(root, manager, { settings, getToolContext: () => context });
		const args = { path: "**/*.txt", limit: 200 };

		// glob's own cut: the listing passed 50 KB, the cut kept over 20 KB, and no artifact holds the rest.
		const own = await new GlobTool(session).execute("glob-own-cut", args);
		const ownCut = truncationOf(own.details);
		expect(ownCut).toBeDefined();
		expect(ownCut?.artifactId).toBeUndefined();
		expect(ownCut?.totalBytes).toBeGreaterThan(50 * 1024);
		expect(Buffer.byteLength(textOf(own))).toBeGreaterThan(20 * 1024);

		// The registry erases each tool's parameter type, as the session's does.
		const glob = wrapToolWithMetaNotice(new GlobTool(session)) as AgentTool;
		const value = asObject(await callSessionTool("glob", args, { session: withTools(session, [glob]) }));

		// The listing glob cut is never presented as complete: no artifact claims to hold the full output.
		expect(value.incomplete).toEqual({ reason: "unreadable" });
		expect(truncationOf(value.details)?.artifactId).toBeUndefined();
		expect(value.text).not.toContain("artifact://");
		expect(value.text.length).toBeLessThan(textOf(own).length);
	});

	describe("(vi) bash notices on restored text", () => {
		/** What the bash tool appends to the output of a command that exits with code 3. */
		const EXIT_3_NOTICES = /^\n\n(Wall time: [\d.]+ seconds\n\n)?Command exited with code 3$/;

		/** A session whose real `bash` tool spills large output into `manager`, as a registered session tool does. */
		function bashSession(manager: SessionManager, overrides: Partial<ToolSession> = {}): ToolSession {
			const session = ownerSession(tempDir, manager, {
				settings: Settings.isolated({
					"async.enabled": false,
					"bash.autoBackground.enabled": false,
					"bashInterceptor.enabled": false,
					"eval.autoProvision": false,
					"python.interpreter": Bun.env.PYTHON ?? $which("python3") ?? $which("python") ?? "",
				}),
				skills: [],
				getClientBridge: () => undefined,
				allocateOutputArtifact: toolType => manager.allocateArtifactPath(toolType),
				...overrides,
			});
			// The registry erases each tool's parameter type, as the session's does.
			return withTools(session, [wrapToolWithMetaNotice(new BashTool(session)) as AgentTool]);
		}

		it("keeps the exit-code notice of a failing command, as on uncut output", async () => {
			const payload = bigJson("failing");
			await Bun.write(path.join(tempDir, "doc.json"), payload);
			const session = bashSession(fileManager());

			const small = asObject(await callSessionTool("bash", { command: "printf hi; exit 3" }, { session }));
			const large = asObject(await callSessionTool("bash", { command: "cat doc.json; exit 3" }, { session }));

			expect(truncationOf(small.details)).toBeUndefined();
			expect(truncationOf(large.details)?.artifactId).toBeDefined();
			expect(small.text.slice(0, 2)).toBe("hi");
			expect(small.text.slice(2)).toMatch(EXIT_3_NOTICES);
			expect(large.text.slice(0, payload.length)).toBe(payload);
			expect(large.text.slice(payload.length)).toMatch(EXIT_3_NOTICES);
			expect(large.hasError).toBe(true);
			expect(isRecord(large.details) ? large.details.exitCode : undefined).toBe(3);
			expect(large.incomplete).toBeUndefined();
		});

		it("keeps the timeout and wall-time notices of a timed-out command", async () => {
			const payload = bigJson("timed-out");
			await Bun.write(path.join(tempDir, "doc.json"), payload);
			const session = bashSession(fileManager());

			const value = asObject(
				await callSessionTool("bash", { command: "cat doc.json; sleep 5", timeout: 1 }, { session }),
			);

			expect(truncationOf(value.details)?.artifactId).toBeDefined();
			const opening = `[Command timed out after 1 seconds]\n${payload}`;
			expect(value.text.slice(0, opening.length)).toBe(opening);
			expect(value.text.slice(opening.length)).toMatch(/\n\nWall time: [\d.]+ seconds$/);
			expect(value.hasError).toBe(true);
			expect(isRecord(value.details) ? value.details.timedOut : undefined).toBe(true);
			expect(value.incomplete).toBeUndefined();
		});

		// Real Python kernels only under PI_PYTHON_INTEGRATION=1, like the other real-kernel suites.
		const PYTHON_INTEGRATION = Bun.env.PI_PYTHON_INTEGRATION === "1";

		/** Each cell prints what `tool.bash` returned for a large output with exit code 3, next to the file it printed. */
		const cells = (file: string) => ({
			js: [
				`const payload = await Bun.file(${JSON.stringify(file)}).text();`,
				'const r = await tool.bash({ command: "cat doc.json; exit 3" });',
				"print(JSON.stringify({ restored: r.text.startsWith(payload), notices: r.text.slice(payload.length),",
				"  hasError: r.hasError, exitCode: r.details.exitCode, incomplete: r.incomplete ?? null }));",
			].join("\n"),
			py: [
				"import json",
				`payload = open(${JSON.stringify(file)}, encoding="utf-8").read()`,
				'r = await tool.bash({"command": "cat doc.json; exit 3"})',
				'print(json.dumps({"restored": r["text"].startswith(payload), "notices": r["text"][len(payload):],',
				'  "hasError": r.get("hasError"), "exitCode": r["details"].get("exitCode"), "incomplete": r.get("incomplete")}))',
			].join("\n"),
		});

		for (const language of ["js", "py"] as const) {
			it.skipIf(language === "py" && !PYTHON_INTEGRATION)(
				`${language}: an eval cell gets the full output and the exit-code notice from tool.bash`,
				async () => {
					const payload = bigJson(`cell-${language}`);
					const file = path.join(tempDir, "doc.json");
					await Bun.write(file, payload);
					const id = `bridge-full-text-${language}-${crypto.randomUUID()}`;
					const session = bashSession(fileManager(), { getEvalSessionId: () => id });

					const result = await new EvalTool(session).execute(`cell-${language}`, {
						language,
						code: cells(file)[language],
					});

					expect(result.isError).not.toBe(true);
					const printed: unknown = JSON.parse(textOf(result));
					if (!isRecord(printed) || typeof printed.notices !== "string") {
						throw new Error("expected the cell's JSON line");
					}
					const { notices, ...rest } = printed;
					expect(rest).toEqual({ restored: true, hasError: true, exitCode: 3, incomplete: null });
					expect(notices).toMatch(EXIT_3_NOTICES);
				},
			);
		}
	});

	it("(vii) returns a page of a paged `read` as complete", async () => {
		const rows = Array.from(
			{ length: 4_000 },
			(_, index) => `row-${String(index + 1).padStart(4, "0")} ${"x".repeat(60)}`,
		);
		await Bun.write(path.join(tempDir, "rows.txt"), `${rows.join("\n")}\n`);
		const session = ownerSession(tempDir, fileManager(), { skills: [] });
		// The registry erases each tool's parameter type, as the session's does.
		withTools(session, [wrapToolWithMetaNotice(new ReadTool(session)) as AgentTool]);

		const range = asObject(await callSessionTool("read", { path: "rows.txt:1-100" }, { session }));
		const firstPage = asObject(await callSessionTool("read", { path: "rows.txt" }, { session }));

		// Both record a window with a continuation: the file goes on after the page.
		for (const page of [range, firstPage]) {
			expect(isRecord(page.details) && isRecord(page.details.meta) && page.details.meta.pagedSource).toBe(true);
			expect(truncationOf(page.details)?.nextOffset).toBeGreaterThan(100);
			expect(page.incomplete).toBeUndefined();
		}
		expect(range.text).toContain("row-0001");
		expect(range.text).toContain("row-0100");
	});

	it("(viii) joins the text blocks of a cut and of an uncut result the same way", async () => {
		const twoBlocks = (name: string, first: string, second: string): AgentTool =>
			wrapToolWithMetaNotice({
				name,
				label: name,
				description: `${name} device`,
				parameters: type({}),
				approval: () => "read",
				async execute() {
					return {
						content: [
							{ type: "text", text: first },
							{ type: "text", text: second },
						],
					};
				},
			} as AgentTool);
		// Two 30 KB blocks: together they pass the 50 KB spill threshold.
		const block = (label: string) =>
			Array.from({ length: 600 }, (_, index) => `${label}-${index} ${"x".repeat(40)}`).join("\n");
		const session = withTools(ownerSession(tempDir, fileManager()), [
			twoBlocks("small_blocks", "first", "second"),
			twoBlocks("big_blocks", block("first"), block("second")),
		]);

		const small = await callSessionTool("small_blocks", {}, { session });
		const big = asObject(await callSessionTool("big_blocks", {}, { session }));

		expect(truncationOf(big.details)?.artifactId).toBeDefined();
		expect(small).toBe("first\nsecond");
		expect(big.text).toBe(`${block("first")}\n${block("second")}`);
	});

	describe("(h) other conversion paths", () => {
		it("restores the text of an eval prelude host call", async () => {
			const payload = bigJson("prelude");
			const manager = memoryManager();
			const prelude: EvalPreludeDefinition = {
				name: "big_prelude",
				documentation: "Returns a large JSON document.",
				javascript: "",
				python: "",
				exports: [],
				approval: "read",
				invoke: (_parameters, context) =>
					postProcessToolResult(
						{ content: [{ type: "text", text: payload }], details: {} },
						"big_prelude",
						context.context,
					),
			};
			const session = ownerSession(tempDir, manager, { getEvalPreludes: () => [prelude] });

			const value = asObject(
				await callSessionTool("__prelude__", { name: "big_prelude", parameters: {} }, { session }),
			);

			expect(value.text).toBe(payload);
		});

		it("restores the text of a JavaScript speculative claim", async () => {
			const payload = bigJson("js-claim");
			const manager = memoryManager();
			const committed = await spilled(manager, "read", payload);
			const read = fixedTool(
				"read",
				{ content: [{ type: "text", text: "not executed" }] },
				type({ path: "string" }),
			);
			const shadowCell = { claim: async () => committed } as unknown as EvalShadowCellSession;
			const session = withTools(ownerSession(tempDir, manager), [read]);

			const value = asObject(
				await callSessionTool(
					"read",
					{ path: "a.txt" },
					{ session, shadowCell, identity: { siteId: "js:0", occurrence: 0 } },
				),
			);

			expect(textOf(committed)).not.toBe(payload);
			expect(value.text).toBe(payload);
		});

		it("restores the text of a Python bridge speculative claim", async () => {
			const payload = bigJson("py-claim");
			const manager = memoryManager();
			const committed = await spilled(manager, "read", payload);
			const read = fixedTool(
				"read",
				{ content: [{ type: "text", text: "not executed" }] },
				type({ path: "string" }),
			);
			const shadowCell = { claim: async () => committed } as unknown as EvalShadowCellSession;
			const session = withTools(ownerSession(tempDir, manager), [read]);
			const info = await ensurePyToolBridge();
			const sessionId = `py-full-text-${crypto.randomUUID()}`;
			const unregister = registerPyToolBridge(sessionId, "run", { toolSession: session, shadowCell });
			try {
				const response = await fetch(`${info.url}/v1/tool`, {
					method: "POST",
					headers: { "Content-Type": "application/json", Authorization: `Bearer ${info.token}` },
					body: JSON.stringify({
						session: sessionId,
						run: "run",
						name: "read",
						args: { path: "a.txt" },
						identity: { siteId: "py:0", occurrence: 0 },
					}),
				});
				const body: unknown = await response.json();
				if (!isRecord(body)) throw new Error("expected a JSON object response");
				expect(body.error).toBeUndefined();
				expect(asObject(body.value).text).toBe(payload);
			} finally {
				unregister();
			}
		});

		it("restores the text of a speculation cell's claimed value and of the result its dependents read", async () => {
			const payload = bigJson("cell");
			const directory = path.join(tempDir, "cell");
			await fs.mkdir(directory);
			const manager = memoryManager();
			const settings = Settings.isolated({
				"eval.autoBackground.enabled": false,
				"images.autoResize": false,
				"tools.speculativeExecution.enabled": true,
			});
			const read = fixedTool(
				"read",
				{ content: [{ type: "text", text: "not executed" }] },
				type({ path: "string" }),
			);
			const session = ownerSession(directory, manager, {
				settings,
				getEvalSessionId: () => "bridge-full-text-cell",
				getToolForEvalBridge: name => (name === "read" ? read : undefined),
				getAdvertisedEvalToolNames: () => ["read"],
			});
			const evalTool = new EvalTool(session);
			await evalTool.execute("warm-full-text", { language: "js", code: "globalThis.shadowWarm = true" });
			const committed = await spilled(manager, "read", payload);
			const admitted: Array<{ args: unknown }> = [];
			const secondAdmission = Promise.withResolvers<void>();
			const coordinator: SpeculativeOperationSink = {
				maxInFlight: 2,
				async admit(definition) {
					admitted.push({ args: definition.toolCall.arguments });
					if (admitted.length === 2) secondAdmission.resolve();
					return {
						candidateId: definition.candidateId,
						fingerprint: "test-fingerprint",
						effect: { kind: "pure" },
						outcome: Promise.resolve({ kind: "result", result: committed, isError: false }),
						commit: async () => committed,
						discard: async () => {},
					};
				},
				close() {},
			};
			const code = 'const a = await tool.read({ path: "a.txt" });\nawait tool.read({ path: a.text + ".txt" });';
			const args = { language: "js", code };
			const shadow = new EvalShadowCellSession({
				coordinator,
				parentToolCallId: "eval-full-text",
				session,
				cwd: directory,
				sessionId: "bridge-full-text-cell",
			});
			try {
				shadow.update(
					{ type: "toolCall", id: "eval-full-text", name: "eval", arguments: args },
					JSON.stringify(args),
				);
				await shadow.finalize({ args });
				expect(admitted).toHaveLength(1);
				const artifactReads = spyOn(manager, "getArtifactContent");

				const value = asObject(
					await shadow.claimValue(
						"read",
						{ path: "a.txt" },
						{ siteId: "js:16", occurrence: 0 },
						Number.MAX_SAFE_INTEGER,
					),
				);
				expect(value.text).toBe(payload);
				// One read serves both the dependents' arguments and the claimed value.
				expect(artifactReads).toHaveBeenCalledTimes(1);

				// The dependent read is admitted with arguments built from the restored text.
				await secondAdmission.promise;
				expect(admitted[1]?.args).toEqual({ path: `${payload}.txt` });
			} finally {
				await shadow.discard("test complete");
			}
		});
	});
});
