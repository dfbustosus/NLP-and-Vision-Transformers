const { OpenAI } = require('openai');

const client = new OpenAI(process.env.OPENAI_API_KEY);

// Get the embedding from OpenAI model
client.embeddings.create({
    input: "black",
    model: "text-embedding-3-small"
}).then(response => {
    const y = response.data[0].embedding;

    // Assuming you have pos and d_model values
    const pos = 1;
    const d_model = 512;

    const pe = Array.from({ length: 1 }, () => Array(d_model).fill(0));
    const pc = Array.from({ length: 1 }, () => Array(d_model).fill(0));

    for (let i = 0; i < d_model; i += 2) {
        pe[0][i] = Math.sin(pos / (10000 ** ((2 * i) / d_model)));
        pc[0][i] = (y[i] * Math.sqrt(d_model)) + pe[0][i];
        pe[0][i + 1] = Math.cos(pos / (10000 ** ((2 * i) / d_model)));
        pc[0][i + 1] = (y[i + 1] * Math.sqrt(d_model)) + pe[0][i + 1];
    }

    console.log(pc);
}).catch(error => {
    console.error('Error:', error);
});
