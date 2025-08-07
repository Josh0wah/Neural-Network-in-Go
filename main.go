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

// neuralNet will contain the information that defines a trained neural network
// Note: w = weight, b = bias
type neuralNet struct {
	config neuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut *mat.Dense
	bOut *mat.Dense
}

// neuralNetConfig definess the neural network's architecture and learning parameters
type neuralNetConfig struct {
	inputNeurons int
	outputNeurons int
	hiddenNeurons int
	numEpochs int
	learningRate float64
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
func (nn *neuralNetwork) train(x, y *mat.Dense) error {
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

	for _, param := range [][]float64 {
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
		addBHidden := func (_, col int, v float64) float64 {
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
	}
}
