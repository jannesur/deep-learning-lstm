console.log("Projekt gestartet");
console.log("TensorFlow.js geladen:", tf);
console.log("Plotly geladen:", Plotly);

let dictionary = [];
let wordToIndex = {};
let indexToWord = {};
let model;
let lossHistory = [];
let lastPredictions = [];
let autoRunning = false;

const sequenceLength = 3;
const epochs = 10;
const batchSize = 8;

function cleanText(text) {
    return text
        .toLowerCase()
        .replace(/[.,!?;:()"]/g, "")
        .replace(/\s+/g, " ")
        .trim();
}

function splitIntoWords(text) {
    const clean = cleanText(text);
    if (clean.length === 0) return [];
    return clean.split(" ");
}

function createDictionary(words) {
    dictionary = [];

    for (const word of words) {
        if (!dictionary.includes(word)) {
            dictionary.push(word);
        }
    }

    wordToIndex = {};
    indexToWord = {};

    for (let i = 0; i < dictionary.length; i++) {
        wordToIndex[dictionary[i]] = i;
        indexToWord[i] = dictionary[i];
    }
}

function createOneHotVector(index) {
    const vector = new Array(dictionary.length).fill(0);
    vector[index] = 1;
    return vector;
}

function createTrainingData(words) {
    const inputs = [];
    const labels = [];

    for (let i = 0; i < words.length - sequenceLength; i++) {
        const inputWords = words.slice(i, i + sequenceLength);
        const targetWord = words[i + sequenceLength];

        const inputSequence = [];

        for (const word of inputWords) {
            inputSequence.push(createOneHotVector(wordToIndex[word]));
        }

        labels.push(createOneHotVector(wordToIndex[targetWord]));
        inputs.push(inputSequence);
    }

    return {
        inputs: tf.tensor3d(inputs),
        labels: tf.tensor2d(labels)
    };
}

function createModel() {
    const newModel = tf.sequential();

    newModel.add(tf.layers.lstm({
        units: 100,
        returnSequences: true,
        inputShape: [sequenceLength, dictionary.length]
    }));

    newModel.add(tf.layers.lstm({
        units: 100
    }));

    newModel.add(tf.layers.dense({
        units: dictionary.length,
        activation: "softmax"
    }));

    newModel.compile({
        optimizer: tf.train.adam(0.001),
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
    });

    return newModel;
}

function plotLossHistory() {
    const epochNumbers = [];

    for (let i = 0; i < lossHistory.length; i++) {
        epochNumbers.push(i + 1);
    }

    Plotly.newPlot(
        "loss-plot",
        [{
            x: epochNumbers,
            y: lossHistory,
            mode: "lines+markers",
            type: "scatter",
            name: "Loss"
        }],
        {
            title: "Trainings-Loss",
            xaxis: { title: "Epoch" },
            yaxis: { title: "Loss" }
        }
    );
}

async function trainModel() {
    const text = document.getElementById("training-text").value;

    if (text.trim().length === 0) {
        alert("Bitte Trainingsdaten eingeben.");
        return;
    }

    const words = splitIntoWords(text);

    if (words.length <= sequenceLength) {
        alert("Der Trainingstext ist zu kurz.");
        return;
    }

    document.getElementById("training-status").textContent =
        "Trainingsdaten werden vorbereitet...";

    await tf.nextFrame();

    createDictionary(words);

    const trainingData = createTrainingData(words);

    if (model !== undefined) {
        model.dispose();
    }

    model = createModel();
    lossHistory = [];
    lastPredictions = [];

    document.getElementById("prediction-output").innerHTML = "";

    document.getElementById("training-status").textContent =
        "Training läuft...";

    await tf.nextFrame();

    await model.fit(
        trainingData.inputs,
        trainingData.labels,
        {
            epochs: epochs,
            batchSize: batchSize,
            shuffle: true,
            callbacks: {
                onEpochEnd: async function(epoch, logs) {
                    lossHistory.push(logs.loss);

                    document.getElementById("training-status").textContent =
                        "Training läuft... Epoche " +
                        (epoch + 1) +
                        " / " +
                        epochs +
                        ", Loss: " +
                        logs.loss.toFixed(4);

                    plotLossHistory();

                    await tf.nextFrame();
                }
            }
        }
    );

    document.getElementById("training-status").textContent =
        "Training abgeschlossen.";

    calculateTopKAccuracy(words);

    trainingData.inputs.dispose();
    trainingData.labels.dispose();

    await tf.nextFrame();
}

function createInputFromPrompt(promptText) {
    const words = splitIntoWords(promptText);

    if (words.length < sequenceLength) {
        return null;
    }

    const lastWords = words.slice(words.length - sequenceLength);
    const inputSequence = [];

    for (const word of lastWords) {
        const index = wordToIndex[word];

        if (index === undefined) {
            return null;
        }

        inputSequence.push(createOneHotVector(index));
    }

    return tf.tensor3d([inputSequence]);
}

async function predictNextWords() {
    if (model === undefined) {
        alert("Bitte zuerst trainieren.");
        return;
    }

    const promptText = document.getElementById("prompt-input").value;

    if (promptText.trim().length === 0) {
        alert("Bitte einen Text eingeben.");
        return;
    }

    const inputTensor = createInputFromPrompt(promptText);

    if (inputTensor === null) {
        alert("Mindestens drei bekannte Wörter eingeben.");
        return;
    }

    const predictionTensor = model.predict(inputTensor);
    const predictionValues = await predictionTensor.data();

    const predictions = [];

    for (let i = 0; i < predictionValues.length; i++) {
        predictions.push({
            word: indexToWord[i],
            probability: predictionValues[i]
        });
    }

    predictions.sort(function(a, b) {
        return b.probability - a.probability;
    });

    lastPredictions = predictions;

    showPredictions(predictions.slice(0, 10));

    inputTensor.dispose();
    predictionTensor.dispose();

    await tf.nextFrame();
}

function showPredictions(predictions) {
    const output = document.getElementById("prediction-output");
    output.innerHTML = "";

    for (const prediction of predictions) {
        const button = document.createElement("button");

        button.className = "prediction-item";

        button.textContent =
            prediction.word +
            " (" +
            (prediction.probability * 100).toFixed(2) +
            "%)";

        button.addEventListener("click", async function() {
            addWordToPrompt(prediction.word);
            await predictNextWords();
        });

        output.appendChild(button);
    }
}

function addWordToPrompt(word) {
    const promptInput = document.getElementById("prompt-input");
    const text = promptInput.value.trim();

    if (text.length === 0) {
        promptInput.value = word;
    } else {
        promptInput.value = text + " " + word;
    }
}

async function acceptBestPrediction() {
    if (lastPredictions.length === 0) {
        await predictNextWords();
    }

    if (lastPredictions.length > 0) {
        const bestWord = lastPredictions[0].word;
        addWordToPrompt(bestWord);
        await predictNextWords();
    }
}

async function acceptWeightedTopPrediction() {
    if (lastPredictions.length === 0) {
        await predictNextWords();
    }

    if (lastPredictions.length === 0) {
        return;
    }

    const topPredictions = lastPredictions.slice(0, 5);

    let probabilitySum = 0;

    for (const prediction of topPredictions) {
        probabilitySum += prediction.probability;
    }

    let randomValue = Math.random() * probabilitySum;
    let selectedWord = topPredictions[0].word;

    for (const prediction of topPredictions) {
        randomValue -= prediction.probability;

        if (randomValue <= 0) {
            selectedWord = prediction.word;
            break;
        }
    }

    addWordToPrompt(selectedWord);
    await predictNextWords();
}

async function startAutoGeneration() {
    if (autoRunning === true) {
        return;
    }

    autoRunning = true;

    for (let i = 0; i < 10; i++) {
        if (autoRunning === false) {
            break;
        }

        await acceptWeightedTopPrediction();

        await new Promise(function(resolve) {
            setTimeout(resolve, 500);
        });

        await tf.nextFrame();
    }

    autoRunning = false;
}

function stopAutoGeneration() {
    autoRunning = false;
}

function resetApplication() {
    autoRunning = false;
    lastPredictions = [];
    dictionary = [];
    wordToIndex = {};
    indexToWord = {};
    lossHistory = [];

    if (model !== undefined) {
        model.dispose();
        model = undefined;
    }

    document.getElementById("prompt-input").value = "";
    document.getElementById("prediction-output").innerHTML = "";
    document.getElementById("training-status").textContent =
        "Modell noch nicht trainiert.";

    document.getElementById("accuracy-1").textContent = "-";
    document.getElementById("accuracy-5").textContent = "-";
    document.getElementById("accuracy-10").textContent = "-";
    document.getElementById("accuracy-20").textContent = "-";
    document.getElementById("accuracy-100").textContent = "-";

    Plotly.purge("loss-plot");
}

function calculateTopKAccuracy(words) {
    let correctTop1 = 0;
    let correctTop5 = 0;
    let correctTop10 = 0;
    let correctTop20 = 0;
    let correctTop100 = 0;
    let total = 0;

    for (let i = 0; i < words.length - sequenceLength; i++) {
        const inputWords = words.slice(i, i + sequenceLength);
        const correctWord = words[i + sequenceLength];
        const prompt = inputWords.join(" ");

        const inputTensor = createInputFromPrompt(prompt);

        if (inputTensor === null) {
            continue;
        }

        const predictionTensor = model.predict(inputTensor);
        const predictionValues = predictionTensor.dataSync();

        const predictions = [];

        for (let j = 0; j < predictionValues.length; j++) {
            predictions.push({
                word: indexToWord[j],
                probability: predictionValues[j]
            });
        }

        predictions.sort(function(a, b) {
            return b.probability - a.probability;
        });

        const top1 = predictions.slice(0, Math.min(1, predictions.length));
        const top5 = predictions.slice(0, Math.min(5, predictions.length));
        const top10 = predictions.slice(0, Math.min(10, predictions.length));
        const top20 = predictions.slice(0, Math.min(20, predictions.length));
        const top100 = predictions.slice(0, Math.min(100, predictions.length));

        if (top1.map(item => item.word).includes(correctWord)) correctTop1++;
        if (top5.map(item => item.word).includes(correctWord)) correctTop5++;
        if (top10.map(item => item.word).includes(correctWord)) correctTop10++;
        if (top20.map(item => item.word).includes(correctWord)) correctTop20++;
        if (top100.map(item => item.word).includes(correctWord)) correctTop100++;

        total++;

        inputTensor.dispose();
        predictionTensor.dispose();
    }

    if (total === 0) {
        document.getElementById("accuracy-1").textContent = "-";
        document.getElementById("accuracy-5").textContent = "-";
        document.getElementById("accuracy-10").textContent = "-";
        document.getElementById("accuracy-20").textContent = "-";
        document.getElementById("accuracy-100").textContent = "-";
        return;
    }

    document.getElementById("accuracy-1").textContent =
        (correctTop1 / total * 100).toFixed(2) + "%";

    document.getElementById("accuracy-5").textContent =
        (correctTop5 / total * 100).toFixed(2) + "%";

    document.getElementById("accuracy-10").textContent =
        (correctTop10 / total * 100).toFixed(2) + "%";

    document.getElementById("accuracy-20").textContent =
        (correctTop20 / total * 100).toFixed(2) + "%";

    document.getElementById("accuracy-100").textContent =
        (correctTop100 / total * 100).toFixed(2) + "%";
}

const trainButton = document.getElementById("train-button");
const predictButton = document.getElementById("predict-button");
const nextButton = document.getElementById("next-button");
const autoButton = document.getElementById("auto-button");
const stopButton = document.getElementById("stop-button");
const resetButton = document.getElementById("reset-button");

trainButton.addEventListener("click", async function() {
    await trainModel();
});

predictButton.addEventListener("click", async function() {
    await predictNextWords();
});

nextButton.addEventListener("click", async function() {
    await acceptBestPrediction();
});

autoButton.addEventListener("click", async function() {
    await startAutoGeneration();
});

stopButton.addEventListener("click", function() {
    stopAutoGeneration();
});

resetButton.addEventListener("click", function() {
    resetApplication();
});