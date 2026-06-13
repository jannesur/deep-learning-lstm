console.log("Projekt gestartet");

console.log("TensorFlow.js geladen:", tf);
console.log("Plotly geladen:", Plotly);

let dictionary = [];

let wordToIndex = {};

let indexToWord = {};


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


// Trainingstext vorbereiten

function prepareTrainingText() {

    const text = document
        .getElementById("prompt-input")
        .value;

    const words = splitIntoWords(text);

    createDictionary(words);

    console.log("Anzahl Wörter:", words.length);

    console.log("Dictionary:", dictionary);

    console.log("wordToIndex:", wordToIndex);

    console.log("indexToWord:", indexToWord);
}


// Buttons

const predictButton =
    document.getElementById("predict-button");

const nextButton =
    document.getElementById("next-button");

const autoButton =
    document.getElementById("auto-button");

const stopButton =
    document.getElementById("stop-button");

const resetButton =
    document.getElementById("reset-button");


// Vorhersage

predictButton.addEventListener(

    "click",

    function () {

        prepareTrainingText();

        console.log(
            "Textvorverarbeitung abgeschlossen"
        );
    }
);


// Platzhalter

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