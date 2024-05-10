import { HfInference } from '@huggingface/inference';
import { config } from 'dotenv';

config({ path: '.env.local' });

const hf = new HfInference(process.env.HF_TOKEN);

/**
 * Calculates the dot product of two arrays.
 *
 * @param a - The first array.
 * @param b - The second array.
 * @returns The dot product of the two arrays.
 * @throws {Error} If the lengths of the input arrays are not equal.
 */
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

const input1 = 'That is a happy person';
const output1 = await hf.featureExtraction({
  model: 'sentence-transformers/all-MiniLM-L6-v2',
  inputs: input1,
});

const input2 = 'That is a happy person';
const output2 = await hf.featureExtraction({
  model: 'sentence-transformers/all-MiniLM-L6-v2',
  inputs: input2,
});

if (is1DArray(output1) && is1DArray(output2)) {
  const similarity = dotProduct(output1, output2);

  console.log('input1:', input1)
  console.log('input2:', input2)
  console.log('Similarity:', similarity)
}

/**
 * Checks if the given value is a one-dimensional array.
 *
 * @param value - The value to check.
 * @returns `true` if the value is a one-dimensional array, `false` otherwise.
 * @template T - The type of elements in the array.
 */
function is1DArray<T>(value: (T | T[] | T[][])[]): value is T[] {
  return !Array.isArray(value[0]);
}

/**
 * Notes:
 * - This connects to Hugging Face's Inference API
 *   - so nothing is hosted locally, we're hitting a web server
 * - Hugging Face calls embeddings "Feature Extraction"
 * - Some models return an embedding for each token of the input
 *   - something to watch out for when expecting a single embedding an input
 *   - Hugging Face sometimes averages the embeddings to return a single embedding
 *     - this seems to happen when the model is tagged with 'Sentence Transformers'
 */
