import { Tensor, pipeline } from '@xenova/transformers';

const generateEmbeddings = await pipeline(
  'feature-extraction',
  'Xenova/all-MiniLM-L6-v2'
);

function dotProduct(a: number[], b: number[]) {
  if (a.length !== b.length) {
    throw new Error('Both arguments must have the same length');
  }

  let result = 0;

  for (let i = 0; i < a.length; i++) {
    result += a[i] * b[i];
  }

  return result;
}

/**
 * Outputs are Tensor objects which contain a data property that is an array of numbers
 */
const output1: Tensor = await generateEmbeddings('That is a happy person', {
  pooling: 'mean', // without this, we get an embedding for each token rather than for the entire input
  normalize: true, // normalization allows us to perform dot product operation
});

const output2: Tensor = await generateEmbeddings('That is a sad person', {
  pooling: 'mean',
  normalize: true,
});

const similarity = dotProduct(output1.data, output2.data);

console.log(similarity);

/**
 * Notes:
 * - This generates an embedding locally via the @xenova/transformers library
 *   - under the hood, Transformers.js fetches the model, 'Xenova/all-MiniLM-L6-v2', from Hugging
 *     Face on the fly at runtime and caches it locally under
 *     `node_modules/@xenova/transformers/.cache/Xenova/all-MiniLM-L6-v2`
 *   - in other words, no reaching out to a web server after that initial fetch
 * - It's worth considering if you want to fetch a model on the fly at runtime or if you'd rather
 *   pre-fetch the model an distribute it with your app
 *   - e.g. could embed the model in your Docker image
 * - Xenova models are run in the Onnx runtime
 *   - as opposed to, for example, the PyTorch runtime
 *   - models are in .onnx files
 * - Models vs quantized models
 *   - quantized models are "compressed" versions of the original models but it reduces the
 *     precision of models
 *   - balance lightweight-ness with precision
 */
