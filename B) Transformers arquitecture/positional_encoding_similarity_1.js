require('dotenv').config();
const { OpenAI } = require('openai');

const openai = new OpenAI(process.env.OPENAI_API_KEY);

// Function to get embeddings of words
async function getEmbeddings(words) {
    const embeddings = [];
    for (const word of words) {
        try {
            const embeddingResponse = await openai.embeddings.create({
                model: "text-embedding-3-small",
                input: word,
                encoding_format: "float",
            });

            // Extracting embedding from the response
            embeddings.push(embeddingResponse.data[0].embedding);
        } catch (error) {
            console.error(`Error while getting embedding for '${word}':`, error.message);
        }
    }
    return embeddings;
}

// Function to calculate cosine similarity
function calculateCosineSimilarity(vectorA, vectorB) {
    const dotProduct = vectorA.reduce((acc, val, i) => acc + val * vectorB[i], 0);
    const normA = Math.sqrt(vectorA.reduce((acc, val) => acc + val * val, 0));
    const normB = Math.sqrt(vectorB.reduce((acc, val) => acc + val * val, 0));
    return dotProduct / (normA * normB);
}

async function main() {
    const words = ["black", "brown"];

    try {
        // Get embeddings for the words
        const embeddings = await getEmbeddings(words);

        // Calculate cosine similarity
        const similarityScore = calculateCosineSimilarity(embeddings[0], embeddings[1]);

        console.log(`Cosine similarity between '${words[0]}' and '${words[1]}': ${similarityScore}`);
    } catch (error) {
        console.error("Error:", error.message);
    }
}

main();
