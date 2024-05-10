import { pipeline } from 'https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.1';

/**
 * Because we're using the `defer` attribute in the <script> tag in index.html, we know that these
 * elements will be accessible by the time this script runs.
 */
const input1 = document.getElementById('input1');
const input2 = document.getElementById('input2');
const generateButton = document.getElementById('generate-button');
const output = document.getElementById('output');

const generateEmbeddings = await pipeline(
  'feature-extraction',
  'Xenova/all-MiniLM-L6-v2'
);

generateButton.disabled = false;

function dotProduct(a, b) {
  if (a.length !== b.length) {
    throw new Error('Both arguments must have the same length');
  }

  let result = 0;

  for (let i = 0; i < a.length; i++) {
    result += a[i] * b[i];
  }

  return result;
}

generateButton.addEventListener('click', async () => {
  const output1 = await generateEmbeddings(input1.value, {
    pooling: 'mean',
    normalize: true,
  });

  const output2 = await generateEmbeddings(input2.value, {
    pooling: 'mean',
    normalize: true,
  });

  const similarity = dotProduct(output1.data, output2.data);

  output.innerText = similarity;
});

/**
 * Notes:
 * - This JS is loaded via a <script> tag in index.html
 *   - the <script> tag uses the `type="module"` attribute to run the script as an ESM module
 *     - allows us to use `import` and `export` statements and other ESM features
 *     - a new-ish browser feature
 *   - the <script> tag uses the `defer` attribute which will make sure the script is run only after
 *     the entire DOM has loaded
 *     - important because we want to make sure the elements we are trying to access are available
 * - We can run this little app using VSCode's Live Server extension, which can spin up a web server
 *   without requiring a fully-fledged FE framework
 * - Again, we need to fetch the model from Hugging Face the very first time we run this, but after
 *   that, Transformers.js will cache the model for us in the browser
 *   - to see this in the cache, go to DevTools > Application > Cache Storage > transformers-cache
 */
