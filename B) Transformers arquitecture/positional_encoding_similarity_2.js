function cosineSimilarity(vectorA, vectorB) {
    const dotProduct = vectorA.reduce((acc, val, i) => acc + val * vectorB[i], 0);
    const normA = Math.sqrt(vectorA.reduce((acc, val) => acc + val * val, 0));
    const normB = Math.sqrt(vectorB.reduce((acc, val) => acc + val * val, 0));
    const similarity = dotProduct / (normA * normB);
    return similarity;
}

function positionalEncoding(pos, pe, d_model = 512) {
    for (let i = 0; i < d_model; i += 2) {
        pe[i] = Math.sin(pos / (10000 ** ((2 * i) / d_model)));
        pe[i + 1] = Math.cos(pos / (10000 ** ((2 * i) / d_model)));
    }
    return pe;
}

// Define d_model
const d_model = 512;

// Initialize positional encoding arrays
const pos_2_output = new Array(d_model).fill(0);
const pos_10_output = new Array(d_model).fill(0);

// Calculate positional encoding for pos = 2
positionalEncoding(2, pos_2_output);

// Calculate positional encoding for pos = 10
positionalEncoding(10, pos_10_output);

// Calculate cosine similarity
const similarityScore = cosineSimilarity(pos_2_output, pos_10_output);
console.log(`Cosine similarity between pos (2) and pos (10): ${similarityScore}`);
