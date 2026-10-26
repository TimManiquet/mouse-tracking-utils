import { CanvasError, createCanvas, joinSession } from "@github/copilot-sdk/extension";
import {
  addUtility,
  connectUtilities,
  getWorkflow,
  openWorkflowCanvas,
  stopWorkflowCanvas,
} from "./workflow.mjs";

const instances = new Map();

function workflowFor(instanceId) {
  const workflow = instances.get(instanceId);
  if (!workflow) {
    throw new CanvasError("instance_not_open", "Open the workflow canvas before using its actions.");
  }
  return workflow;
}

const canvas = createCanvas({
  id: "mouse-tracking-utils-workflow",
  displayName: "Mouse-tracking toolbox workflow",
  description: "Map the toolbox utilities and their input, output, and dependency flow.",
  actions: [
    {
      name: "get_workflow",
      description: "Return the current toolbox workflow map.",
      handler: ({ instanceId }) => getWorkflow(workflowFor(instanceId)),
    },
    {
      name: "add_utility",
      description: "Add a utility or data step to the workflow map.",
      inputSchema: {
        type: "object",
        properties: {
          label: { type: "string", minLength: 1, maxLength: 80 },
          details: { type: "string", maxLength: 240 },
          category: { type: "string", enum: ["input", "utility", "data", "output"] },
        },
        required: ["label"],
        additionalProperties: false,
      },
      handler: ({ instanceId, input }) => addUtility(workflowFor(instanceId), input),
    },
    {
      name: "connect_utilities",
      description: "Add a directed dependency arrow between two workflow nodes.",
      inputSchema: {
        type: "object",
        properties: {
          from: { type: "string", minLength: 1, maxLength: 80 },
          to: { type: "string", minLength: 1, maxLength: 80 },
          label: { type: "string", maxLength: 80 },
        },
        required: ["from", "to"],
        additionalProperties: false,
      },
      handler: ({ instanceId, input }) => connectUtilities(workflowFor(instanceId), input),
    },
  ],
  open: async (ctx) => {
    try {
      const instance = await openWorkflowCanvas(ctx.instanceId, ctx.session?.workingDirectory);
      instances.set(ctx.instanceId, instance);
      return { url: instance.url, title: "Mouse-tracking toolbox workflow" };
    } catch (error) {
      if (error instanceof CanvasError) throw error;
      throw new CanvasError("workflow_open_failed", `Could not open workflow canvas: ${error.message}`);
    }
  },
  onClose: ({ instanceId }) => {
    stopWorkflowCanvas(instanceId);
    instances.delete(instanceId);
  },
});

await joinSession({ canvases: [canvas] });
