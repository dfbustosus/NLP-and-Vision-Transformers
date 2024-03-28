
const math = require('mathjs');
const { OpenAI } = require('openai');
const client = new OpenAI(process.env.OPENAI_API_KEY);

// Function to calculate cosine similarity
function cosineSimilarity(vector_a, vector_b) {
    const dotProduct = math.dot(vector_a, vector_b);
    const norm_a = math.norm(vector_a);
    const norm_b = math.norm(vector_b);
    const similarity = dotProduct / (norm_a * norm_b);
    return similarity;
}

// Get the embeddings for both words
const words = ["black", "brown"];
const embeddings = [];

async function getEmbeddings() {
    for (const word of words) {
        const response = await client.embeddings.create({
            input: word,
            model: "text-embedding-3-small"
        });
        embeddings.push(response.data[0].embedding);
    }

    // Assuming you have pos and d_model values
    const pos = 1;
    const d_model = 512;

    // Calculate the pc values for both words
    const pcs = [];

    for (const embedding of embeddings) {
        const pe = math.zeros(512);
        const pc = math.zeros(512);
        for (let i = 0; i < 512; i += 2) {
            pe.set([i], math.sin(pos / (10000 ** ((2 * i) / d_model))));
            pc.set([i], (embedding[i] * math.sqrt(d_model)) + pe.get([i]));
            pe.set([i + 1], math.cos(pos / (10000 ** ((2 * (i + 1)) / d_model))));
            pc.set([i + 1], (embedding[i + 1] * math.sqrt(d_model)) + pe.get([i + 1]));
        }
        pcs.push(pc);
    }

    // Calculate cosine similarity between pc values of "black" and "brown"
    const similarity = cosineSimilarity(math.flatten(pcs[0]), math.flatten(pcs[1]));
    console.log("Cosine similarity between pc(black) and pc(brown):", similarity);
}

getEmbeddings();