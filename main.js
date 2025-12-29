let session;

async function loadModel() {
  session = await ort.InferenceSession.create(
    "model/model.onnx",
    {
      executionProviders: ["wasm"],
    }
  );
  console.log("ONNX model loaded 🚀");
}

async function runAI() {
  if (!session) await loadModel();

  // Example input (replace with your real model input)
  const inputData = new Float32Array([1, 2, 3, 4]);
  const inputTensor = new ort.Tensor("float32", inputData, [1, 4]);

  // IMPORTANT: input name must match the ONNX model input name
  const results = await session.run({ input: inputTensor });

  console.log("AI Output:", results);
}

loadModel();
