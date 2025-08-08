// This simple neural network contains a single hidden layer, and will be trained with the famous
// Iris Flower Data Set to distinguish between 3 species of Iris flowers.
// https://en.wikipedia.org/wiki/Iris_flower_data_set

package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// neuralNet contains all of the information
// that defines a trained neural network.
type neuralNet struct {
	config  neuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

// neuralNetConfig defines our neural network
// architecture and learning parameters.
type neuralNetConfig struct {
	inputNeurons  int
	outputNeurons int
	hiddenNeurons int
	numEpochs     int
	learningRate  float64
}

func main() {
	// Form training Matrices
	inputs, labels := makeInputsAndLabels("data/train.csv")

	// Define network arrchitecture and learning parameters
	config := neuralNetConfig{
		inputNeurons:  4,
		outputNeurons: 3,
		hiddenNeurons: 3,
		numEpochs:     5000,
		learningRate:  0.3,
	}

	// Train neural network
	network := newNetwork(config)
	if err := network.train(inputs, labels); err != nil {
		log.Fatal(err)
	}

	// Form the testing matrices
	testInputs, testLabels := makeInputsAndLabels("data/train.csv")

	// Use trained model  to make predictions
	predictions, err := network.predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}

	// Calculate accuracy of the model
	var truePosNeg int
	numPreds, _ := predictions.Dims()
	for i := 0; i < numPreds; i++ {
		// Get Label
		labelRow := mat.Row(nil, i, testLabels)
		var prediction int
		for idx, label := range labelRow {
			if label == 1.0 {
				prediction = idx
				break
			}
		}

		// Accumulate true positive/negative count
		if predictions.At(i, prediction) == floats.Max(mat.Row(nil, i, predictions)) {
			truePosNeg++
		}
	}

	// Calculate accuacy
	accuracy := float64(truePosNeg) / float64(numPreds)

	// Output accuracy
	fmt.Printf("\nAccuracy  = %0.2f\n\n", accuracy)
}

// newNetwork initializes a new neural network
func newNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}

// The Sigmoid function and it's derivitive will be used for acctivation and backpropagatian respectively
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

func sigmoidPrime(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

// train trains a neural network using backpropagation
func (nn *neuralNet) train(x, y *mat.Dense) error {
	// init b and w
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
	bHidden := mat.NewDense(1, nn.config.hiddenNeurons, nil)
	wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
	bOut := mat.NewDense(1, nn.config.outputNeurons, nil)

	wHiddenRaw := wHidden.RawMatrix().Data
	bHiddenRaw := bHidden.RawMatrix().Data
	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data

	for _, param := range [][]float64{
		wHiddenRaw,
		bHiddenRaw,
		wOutRaw,
		bOutRaw,
	} {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	//Define the output of the neural network
	output := new(mat.Dense)

	// Adjust w and b with back propagation
	if err := nn.backpropagate(x, y, wHidden, bHidden, wOut, bOut, output); err != nil {
		return err
	}

	// Define trained neural netwerk
	nn.wHidden = wHidden
	nn.bHidden = bHidden
	nn.wOut = wOut
	nn.bOut = bOut

	return nil
}

// backpropagate handles the backpropagaion of data
func (nn *neuralNet) backpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {
	// Loop over epochs and backpropagate to train the model
	for i := 0; i < nn.config.numEpochs; i++ {
		hiddenLayerInput := new(mat.Dense)
		hiddenLayerInput.Mul(x, wHidden)
		addBHidden := func(_, col int, v float64) float64 {
			return v + bHidden.At(0, col)
		}
		hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

		hiddenLayerActivations := new(mat.Dense)
		applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
		hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

		outputLayerInput := new(mat.Dense)
		outputLayerInput.Mul(hiddenLayerActivations, wOut)
		addBOut := func(_, col int, v float64) float64 {
			return v + bOut.At(0, col)
		}
		outputLayerInput.Apply(addBOut, outputLayerInput)
		output.Apply(applySigmoid, outputLayerInput)

		// Complete backpropagation
		networkError := new(mat.Dense)
		networkError.Sub(y, output)

		slopeOutputLayer := new(mat.Dense)
		applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrime(v) }
		slopeOutputLayer.Apply(applySigmoidPrime, output)
		slopeHiddenLayer := new(mat.Dense)
		slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

		dOutput := new(mat.Dense)
		dOutput.MulElem(networkError, slopeOutputLayer)
		errorAtHiddenLayer := new(mat.Dense)
		errorAtHiddenLayer.Mul(dOutput, wOut.T())

		dHiddenLayer := new(mat.Dense)
		dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

		//Adjust parameters
		wOutAdj := new(mat.Dense)
		wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
		wOutAdj.Scale(nn.config.learningRate, wOutAdj)
		wOut.Add(wOut, wOutAdj)

		bOutAdj, err := sumAlongAxis(0, dOutput)
		if err != nil {
			return err
		}
		bOutAdj.Scale(nn.config.learningRate, bOutAdj)
		bOut.Add(bOut, bOutAdj)

		wHiddenAdj := new(mat.Dense)
		wHiddenAdj.Mul(x.T(), dHiddenLayer)
		wHiddenAdj.Scale(nn.config.learningRate, wHiddenAdj)
		wHidden.Add(wHidden, wHiddenAdj)

		bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
		if err != nil {
			return err
		}
		bHiddenAdj.Scale(nn.config.learningRate, bHiddenAdj)
		bHidden.Add(bHidden, bHiddenAdj)
	}

	return nil
}

// sumAlongAxis sums a matrix along a particular dimension, preserving the other dimension
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("invalid axis, must be 0 or 1")
	}

	return output, nil
}

// predict makes a prediction basedon a trained neural network
func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {
	// Ensure that neuralNe is a trained model
	if nn.wHidden == nil || nn.wOut == nil {
		return nil, errors.New("Given weights are empty")
	}
	if nn.bHidden == nil || nn.bOut == nil {
		return nil, errors.New("Given weights are empty")
	}

	// Define output of neural network
	output := new(mat.Dense)

	// Complete feed forward
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.wHidden)
	addBHidden := func(_, col int, v float64) float64 {
		return v + nn.bHidden.At(0, col)
	}
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)
	addBOut := func(_, col int, v float64) float64 {
		return v + nn.bOut.At(0, col)
	}
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}

func makeInputsAndLabels(fileName string) (*mat.Dense, *mat.Dense) {
	// Open training data
	f, err := os.Open("fileName")
	if err != nil {
		log.Fatal("err")
	}
	defer f.Close()

	// Create new CSV reader
	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 7

	// Read in CSV records
	rawCSVData, err := reader.ReadAll()
	if err != nil {
		log.Fatal(err)
	}

	// inputsData and labelsData will hold float values used to form matrices later
	inputsData := make([]float64, 4*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))

	// inputsIndex will track the currunt indek of input matrix values
	var inputsIndex int
	var labelsIndex int

	// Move rows into a slice of floats
	for idx, record := range rawCSVData {
		if idx == 0 {
			continue
		}

		// Loop over float columns
		for i, val := range record {
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Add value to labelsData if relevant
			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Add value to slice
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}

	// Form matrices
	inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)
	return inputs, labels
}
