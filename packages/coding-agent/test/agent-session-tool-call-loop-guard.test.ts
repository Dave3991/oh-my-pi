import { afterEach, beforeEach, describe, expect, it } from "bun:test";
import * as path from "node:path";
import { type } from "@oh-my-pi/omptype";
import { Agent, type AgentTool } from "@oh-my-pi/pi-agent-core";
import type { AssistantMessage, Context } from "@oh-my-pi/pi-ai";
import { createMockModel } from "@oh-my-pi/pi-ai/providers/mock";
import { AssistantMessageEventStream } from "@oh-my-pi/pi-ai/utils/event-stream";
import { ModelRegistry } from "@oh-my-pi/pi-coding-agent/config/model-registry";
import { Settings } from "@oh-my-pi/pi-coding-agent/config/settings";
import { AgentSession } from "@oh-my-pi/pi-coding-agent/session/agent-session";
import { AuthStorage } from "@oh-my-pi/pi-coding-agent/session/auth-storage";
import { type CustomMessage, convertToLlm } from "@oh-my-pi/pi-coding-agent/session/messages";
import { SessionManager } from "@oh-my-pi/pi-coding-agent/session/session-manager";
import { TempDir } from "@oh-my-pi/pi-utils";

const zeroUsage = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
} satisfies AssistantMessage["usage"];

describe("AgentSession tool-call loop guard", () => {
	let tempDir: TempDir;
	let authStorage: AuthStorage;
	let session: AgentSession | undefined;

	beforeEach(async () => {
		tempDir = TempDir.createSync("@pi-tool-call-loop-guard-");
		authStorage = await AuthStorage.create(path.join(tempDir.path(), "auth.db"));
		authStorage.keys.setRuntime("openai", "openai-test-key");
	});

	afterEach(async () => {
		await session?.dispose();
		authStorage.close();
		tempDir.removeSync();
	});

	it("injects a hidden redirect before the next model call", async () => {
		const model = createMockModel({ provider: "openai", id: "gpt-test" }).model;
		const modelRegistry = new ModelRegistry(authStorage);
		const contexts: Context[] = [];
		const bashTool: AgentTool = {
			name: "bash",
			label: "Bash",
			description: "Mock bash tool",
			parameters: type({ "command?": "string" }),
			execute: async () => ({ content: [{ type: "text" as const, text: "1263 passed, 4 skipped" }] }),
		};
		let callCount = 0;
		const agent = new Agent({
			getApiKey: () => "test-key",
			initialState: { model, systemPrompt: ["Test"], tools: [bashTool], messages: [] },
			convertToLlm,
			streamFn: (_model, context) => {
				contexts.push(context);
				const toolCallTurn = callCount < 2;
				const toolCallId = `tc-${callCount}`;
				callCount++;
				const message: AssistantMessage = toolCallTurn
					? {
							role: "assistant",
							content: [{ type: "toolCall", id: toolCallId, name: "bash", arguments: { command: "pytest -q" } }],
							api: model.api,
							provider: model.provider,
							model: model.id,
							usage: zeroUsage,
							stopReason: "toolUse",
							timestamp: Date.now(),
						}
					: {
							role: "assistant",
							content: [{ type: "text", text: "Stopped repeating." }],
							api: model.api,
							provider: model.provider,
							model: model.id,
							usage: zeroUsage,
							stopReason: "stop",
							timestamp: Date.now(),
						};
				const stream = new AssistantMessageEventStream();
				queueMicrotask(() => {
					stream.push({ type: "start", partial: message });
					stream.push({ type: "done", reason: toolCallTurn ? "toolUse" : "stop", message });
				});
				return stream;
			},
		});
		const settings = Settings.isolated({
			"compaction.enabled": false,
			"todo.enabled": false,
			"model.toolCallLoopGuard.enabled": true,
			"model.toolCallLoopGuard.threshold": 2,
			"model.toolCallLoopGuard.exemptTools": ["wait"],
		});
		settings.setModelRole("default", `${model.provider}/${model.id}`);
		session = new AgentSession({
			agent,
			sessionManager: SessionManager.inMemory(tempDir.path()),
			settings,
			modelRegistry,
			toolRegistry: new Map([[bashTool.name, bashTool]]),
		});

		await session.prompt("run checks");
		await session.waitForIdle();

		expect(contexts).toHaveLength(3);
		expect(JSON.stringify(contexts[2]!.messages)).toContain("tool_call_loop_detected");
		expect(JSON.stringify(contexts[2]!.messages)).toContain("1263 passed, 4 skipped");
		const redirects = session.agent.state.messages.filter(
			(message): message is CustomMessage =>
				message.role === "custom" && message.customType === "tool-call-loop-redirect",
		);
		expect(redirects).toHaveLength(1);
		expect(redirects[0]!.display).toBe(false);
	});

	describe("xd:// device writes", () => {
		// The model repeats one `write` to an MCP device, as an agent does while it long-polls
		// `xd://mcp__peers_wait`; returns how many loop redirects the session injected.
		async function redirectsForRepeatedDeviceWrite(exemptTools: string[]): Promise<number> {
			const model = createMockModel({ provider: "openai", id: "gpt-test" }).model;
			const writeTool: AgentTool = {
				name: "write",
				label: "Write",
				description: "Mock write tool",
				parameters: type({ "path?": "string", "content?": "string" }),
				execute: async () => ({
					content: [{ type: "text" as const, text: '{"cursor":1,"items":[],"timed_out":true}' }],
				}),
			};
			let callCount = 0;
			const agent = new Agent({
				getApiKey: () => "test-key",
				initialState: { model, systemPrompt: ["Test"], tools: [writeTool], messages: [] },
				convertToLlm,
				streamFn: () => {
					const toolCallTurn = callCount < 2;
					const toolCallId = `tc-${callCount}`;
					callCount++;
					const message: AssistantMessage = toolCallTurn
						? {
								role: "assistant",
								content: [
									{
										type: "toolCall",
										id: toolCallId,
										name: "write",
										arguments: { path: "xd://mcp__peers_wait", content: "{}" },
									},
								],
								api: model.api,
								provider: model.provider,
								model: model.id,
								usage: zeroUsage,
								stopReason: "toolUse",
								timestamp: Date.now(),
							}
						: {
								role: "assistant",
								content: [{ type: "text", text: "Review arrived." }],
								api: model.api,
								provider: model.provider,
								model: model.id,
								usage: zeroUsage,
								stopReason: "stop",
								timestamp: Date.now(),
							};
					const stream = new AssistantMessageEventStream();
					queueMicrotask(() => {
						stream.push({ type: "start", partial: message });
						stream.push({ type: "done", reason: toolCallTurn ? "toolUse" : "stop", message });
					});
					return stream;
				},
			});
			const settings = Settings.isolated({
				"compaction.enabled": false,
				"todo.enabled": false,
				"model.toolCallLoopGuard.enabled": true,
				"model.toolCallLoopGuard.threshold": 2,
				"model.toolCallLoopGuard.exemptTools": exemptTools,
			});
			settings.setModelRole("default", `${model.provider}/${model.id}`);
			session = new AgentSession({
				agent,
				sessionManager: SessionManager.inMemory(tempDir.path()),
				settings,
				modelRegistry: new ModelRegistry(authStorage),
				toolRegistry: new Map([[writeTool.name, writeTool]]),
			});

			await session.prompt("wait for the review");
			await session.waitForIdle();

			return session.agent.state.messages.filter(
				(message): message is CustomMessage =>
					message.role === "custom" && message.customType === "tool-call-loop-redirect",
			).length;
		}

		it("lets a write to a device named in exemptTools repeat", async () => {
			expect(await redirectsForRepeatedDeviceWrite(["wait", "mcp__peers_wait"])).toBe(0);
		});

		it("still redirects a repeated device write when neither the device nor write is exempt", async () => {
			expect(await redirectsForRepeatedDeviceWrite(["wait"])).toBe(1);
		});

		it("keeps exempting device writes when the write tool itself is exempt", async () => {
			expect(await redirectsForRepeatedDeviceWrite(["wait", "write"])).toBe(0);
		});
	});
});
