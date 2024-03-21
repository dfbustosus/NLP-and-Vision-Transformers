require('dotenv').config();
const { OpenAI } = require('openai');

const openai = new OpenAI(process.env.OPENAI_API_KEY);

async function main() {
  try {
    const embeddingResponse = await openai.embeddings.create({
      model: "text-embedding-3-small",
      input: "Your text string goes here",
      encoding_format: "float",
    });

    // Extracting embedding from the response
    const embedding = embeddingResponse.data[0].embedding;

    // Printing the embedding values as a list
    console.log("Embedding:");
    console.log(embedding.join("\n"));
  } catch (error) {
    console.error("Error:", error.message);
  }
}

main();