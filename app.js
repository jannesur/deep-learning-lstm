console.log("Projekt gestartet");

console.log("TensorFlow.js geladen:", tf);

console.log("Plotly geladen:", Plotly);

let dictionary = [];

let wordToIndex = {};

let indexToWord = {};

let model;

let lossHistory = [];

const sequenceLength = 3;


// Text bereinigen

function cleanText(text) {

    return text
        .toLowerCase()
        .replace(/[.,!?;:()"]/g, "")
        .replace(/\s+/g, " ")
        .trim();
}


// Wörter erzeugen

function splitIntoWords(text) {

    const clean = cleanText(text);

    if (clean.length === 0) {

        return [];
    }

    return clean.split(" ");
}


// Dictionary erzeugen

function createDictionary(words) {

    dictionary = [];

    for (const word of words) {

        if (dictionary.includes(word) === false) {

            dictionary.push(word);
        }
    }

    wordToIndex = {};

    indexToWord = {};

    for (let i = 0; i < dictionary.length; i++) {

        const word = dictionary[i];

        wordToIndex[word] = i;

        indexToWord[i] = word;
    }
}


// One-Hot-Vektor erzeugen

function createOneHotVector(index) {

    const vector = [];

    for (let i = 0; i < dictionary.length; i++) {

        if (i === index) {

            vector.push(1);

        } else {

            vector.push(0);
        }
    }

    return vector;
}


// Trainingsdaten erzeugen

function createTrainingData(words) {

    const inputs = [];

    const labels = [];

    for (
        let i = 0;
        i < words.length - sequenceLength;
        i++
    ) {

        const inputWords =
            words.slice(
                i,
                i + sequenceLength
            );

        const targetWord =
            words[i + sequenceLength];

        const inputSequence = [];

        for (const word of inputWords) {

            const index =
                wordToIndex[word];

            inputSequence.push(
                createOneHotVector(index)
            );
        }

        const targetIndex =
            wordToIndex[targetWord];

        const targetVector =
            createOneHotVector(targetIndex);

        inputs.push(inputSequence);

        labels.push(targetVector);
    }

    return {

        inputs: tf.tensor3d(inputs),

        labels: tf.tensor2d(labels)
    };
}


// Modell erzeugen

function createModel() {

    const newModel = tf.sequential();

    newModel.add(

        tf.layers.lstm({

            units: 100,

            returnSequences: true,

            inputShape: [
                sequenceLength,
                dictionary.length
            ]
        })
    );

    newModel.add(

        tf.layers.lstm({

            units: 100
        })
    );

    newModel.add(

        tf.layers.dense({

            units: dictionary.length,

            activation: "softmax"
        })
    );

    newModel.compile({

        optimizer: tf.train.adam(0.001),

        loss: "categoricalCrossentropy",

        metrics: ["accuracy"]
    });

    return newModel;
}


// Loss-Verlauf plotten

function plotLossHistory() {

    const epochs = [];

    for (let i = 0; i < lossHistory.length; i++) {

        epochs.push(i + 1);
    }

    Plotly.newPlot(

        "loss-plot",

        [
            {
                x: epochs,
                y: lossHistory,
                mode: "lines",
                type: "scatter",
                name: "Loss"
            }
        ],

        {
            title: "Trainings-Loss",

            xaxis: {
                title: "Epoch"
            },

            yaxis: {
                title: "Loss"
            }
        }
    );
}


// Modell trainieren

async function trainModel() {

    const text =
        document
            .getElementById("training-text")
            .value;

    const words =
        splitIntoWords(text);

    if (words.length <= sequenceLength) {

        alert(
            "Der Trainingstext ist zu kurz."
        );

        return;
    }

    createDictionary(words);

    const trainingData =
        createTrainingData(words);

    model = createModel();

    lossHistory = [];

    document.getElementById(
        "training-status"
    ).textContent =
        "Training läuft...";

    await model.fit(

        trainingData.inputs,

        trainingData.labels,

        {

            epochs: 20,

            batchSize: 32,

            shuffle: true,

            callbacks: {

                onEpochEnd: function(epoch, logs) {

                    lossHistory.push(
                        logs.loss
                    );

                    console.log(
                        "Epoch:",
                        epoch + 1,
                        "Loss:",
                        logs.loss
                    );
                }
            }
        }
    );

    document.getElementById(
        "training-status"
    ).textContent =
        "Training abgeschlossen.";

    plotLossHistory();

    console.log(
        "Training abgeschlossen"
    );

    trainingData.inputs.dispose();

    trainingData.labels.dispose();
}


// Buttons

const trainButton =
    document.getElementById(
        "train-button"
    );

const predictButton =
    document.getElementById(
        "predict-button"
    );

const nextButton =
    document.getElementById(
        "next-button"
    );

const autoButton =
    document.getElementById(
        "auto-button"
    );

const stopButton =
    document.getElementById(
        "stop-button"
    );

const resetButton =
    document.getElementById(
        "reset-button"
    );


trainButton.addEventListener(

    "click",

    async function () {

        await trainModel();
    }
);


predictButton.addEventListener(

    "click",

    function () {

        console.log(
            "Vorhersage folgt später."
        );
    }
);


nextButton.addEventListener(

    "click",

    function () {

        console.log("Weiter");
    }
);


autoButton.addEventListener(

    "click",

    function () {

        console.log("Auto");
    }
);


stopButton.addEventListener(

    "click",

    function () {

        console.log("Stopp");
    }
);


resetButton.addEventListener(

    "click",

    function () {

        console.log("Reset");
    }
);