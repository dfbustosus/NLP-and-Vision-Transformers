const { sin, cos } = require('mathjs');
const plot = require('nodeplotlib');

// Define the function
function func(x) {
    return sin(2 / (Math.pow(10000, (2 * x / 512))));
}

function func_2(x) {
    return cos(2 / (Math.pow(10000, (2 * x / 512))));
}

// Define the range of x values
const linspace = (start, stop, num) => {
    const step = (stop - start) / (num - 1);
    return Array.from({ length: num }, (_, i) => start + step * i);
}

const x_values = linspace(-100, 100, 1000);

// Compute the corresponding y values
const y_values = x_values.map(x => func(x));
const y_values_2 = x_values.map(x => func_2(x));

// Plot the function
const data = [
    { x: x_values, y: y_values, name: 'y = sin(2 / (10000^(2x/512)))' },
    { x: x_values, y: y_values_2, name: 'y = cos(2 / (10000^(2x/512)))' }
];

// Set the layout options
const layout = {
    title: 'Plot of the Function y = sin(2 / (10000^(2x/512)))',
    xaxis: { title: 'x' },
    yaxis: { title: 'y' },
    showlegend: true,
    grid: { visible: true }
};

// Plot the data with layout
plot.plot(data, layout);
