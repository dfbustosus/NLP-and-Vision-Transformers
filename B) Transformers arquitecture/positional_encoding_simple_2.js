const { sin, cos } = require('mathjs');

function positionalEncoding(pos, d_model = 512) {
    const pe = new Array(1).fill(0).map(() => new Array(d_model).fill(0));
    for (let i = 0; i < d_model; i += 2) {
        pe[0][i] = sin(pos / (10000 ** ((2 * i) / d_model)));
        pe[0][i + 1] = cos(pos / (10000 ** ((2 * i) / d_model)));
    }
    return pe;
}

// Define d_model
const d_model = 512;

// Calculate positional encoding for pos = 2
const pos_2_output = positionalEncoding(2, d_model);

// Calculate positional encoding for pos = 10
const pos_10_output = positionalEncoding(10, d_model);

// Print the outputs
console.log("Positional encoding for pos = 2:");
console.log(pos_2_output);

console.log("\nPositional encoding for pos = 10:");
console.log(pos_10_output);
