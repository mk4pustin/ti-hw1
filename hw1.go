package main

import (
	"fmt"
	"math"
	"math/big"
	"strconv"
	"strings"
)

type Ensemble struct {
	Probabilities [][]float64
	X             int
	Y             int
}

const thousandForRounding = 1000

func main() {
	ensemble := readEnsemblePiece()
	pX, pY := calculateProbabilities(ensemble)

	areIndependent := areIndependent(ensemble, pX, pY)
	if areIndependent {
		fmt.Println("Ансамбли X и Y независимы")
	} else {
		fmt.Println("Ансамбли X и Y зависимы")
	}

	printFloatArr(pX, "p(xi)")
	printFloatArr(pY, "p(yi)")

	var pXGivenY, pYGivenX [][]string
	if !areIndependent {
		pXGivenY, pYGivenX = calculateConditionalProbabilities(ensemble, pX, pY)
	} else {
		pXGivenY = fillConditionalProbabilities(pX, len(pY))
		pYGivenX = fillConditionalProbabilities(pY, len(pX))
	}

	printConditionalProbabilities(pXGivenY, "p(xi | yj)", true)
	printConditionalProbabilities(pYGivenX, "p(yj | xi)", false)

	hX := entropy(pX)
	hY := entropy(pY)
	fmt.Println("H(X) = ", hX)
	fmt.Println("H(Y) = ", hY)

	hXY := jointEntropy(ensemble.Probabilities)
	fmt.Println("H(XY) = ", hXY)

	hYGivenX := conditionalEntropy(pYGivenX, pX)
	hXGivenY := conditionalEntropy(pXGivenY, pY)
	fmt.Println("Hx(Y) = ", hYGivenX)
	fmt.Println("Hy(X) = ", hXGivenY)
}

func readEnsemblePiece() Ensemble {
	var xNum, yNum int

	fmt.Print("Введите количество элементов ансамбля X: ")
	fmt.Scan(&xNum)

	fmt.Print("Введите количество элементов ансамбля Y: ")
	fmt.Scan(&yNum)

	probabilities := make([][]float64, xNum)

	for i := 0; i < xNum; i++ {
		probabilities[i] = make([]float64, yNum)
		for j := 0; j < yNum; j++ {
			fmt.Printf("Введите вероятность для x%dy%d в формате 0.0: ", i+1, j+1)
			fmt.Scan(&probabilities[i][j])
		}
	}

	return Ensemble{
		X:             xNum,
		Y:             yNum,
		Probabilities: probabilities,
	}
}

func calculateProbabilities(ensemble Ensemble) ([]float64, []float64) {
	pX := make([]float64, len(ensemble.Probabilities))
	pY := make([]float64, len(ensemble.Probabilities[0]))

	for i := 0; i < len(ensemble.Probabilities); i++ {
		for j := 0; j < len(ensemble.Probabilities[i]); j++ {
			pX[i] += ensemble.Probabilities[i][j]
			pY[j] += ensemble.Probabilities[i][j]
		}
	}

	return pX, pY
}

func areIndependent(ensemble Ensemble, pX []float64, pY []float64) bool {
	for i := 0; i < len(pX); i++ {
		for j := 0; j < len(pY); j++ {
			curValue := round(ensemble.Probabilities[i][j])
			reqValue := round(pX[i] * pY[j])
			if curValue != reqValue {
				return false
			}
		}
	}

	return true
}

func printFloatArr(arr []float64, name string) {
	fmt.Print(name + ": ")
	for i := 0; i < len(arr); i++ {
		fmt.Print(round(arr[i]), " ")
	}
	fmt.Println()
}

func printStringArr(arr []string, name string) {
	fmt.Print(name + ": ")
	for i := 0; i < len(arr); i++ {
		fmt.Print(arr[i], " ")
	}
	fmt.Println()
}

func printConditionalProbabilities(probabilities [][]string, name string, isGivenX bool) {
	fmt.Println(name + ": ")
	for i := 0; i < len(probabilities); i++ {
		var rowName string
		if isGivenX {
			rowName = strings.Replace(name, "i", strconv.Itoa(i+1), -1)
		} else {
			rowName = strings.Replace(name, "j", strconv.Itoa(i+1), -1)
		}
		printStringArr(probabilities[i], rowName)
	}
}

func calculateConditionalProbabilities(ensemble Ensemble, pX []float64, pY []float64) ([][]string, [][]string) {
	pXGivenY := make([][]string, len(pX))
	pYGivenX := make([][]string, len(pY))

	for i := 0; i < len(pX); i++ {
		pXGivenY[i] = make([]string, len(pY))
		for j := 0; j < len(pY); j++ {
			numerator := big.NewInt(int64(ensemble.Probabilities[i][j] * thousandForRounding))
			denominator := big.NewInt(int64(pY[j] * thousandForRounding))

			gcd := new(big.Int).GCD(nil, nil, numerator, denominator)

			numerator.Div(numerator, gcd)
			denominator.Div(denominator, gcd)

			fraction := fmt.Sprintf("%s/%s", numerator.String(), denominator.String())
			pXGivenY[i][j] = fraction
		}
	}

	for j := 0; j < len(pY); j++ {
		pYGivenX[j] = make([]string, len(pX))
		for i := 0; i < len(pX); i++ {
			numerator := big.NewInt(int64(ensemble.Probabilities[i][j] * thousandForRounding))
			denominator := big.NewInt(int64(pX[i] * thousandForRounding))

			gcd := new(big.Int).GCD(nil, nil, numerator, denominator)

			numerator.Div(numerator, gcd)
			denominator.Div(denominator, gcd)

			fraction := fmt.Sprintf("%s/%s", numerator.String(), denominator.String())
			pYGivenX[j][i] = fraction
		}
	}

	return pXGivenY, pYGivenX
}

func fillConditionalProbabilities(p []float64, size int) [][]string {
	pGiven := make([][]string, len(p))
	for i := 0; i < len(p); i++ {
		pGiven[i] = make([]string, size)
		for j := 0; j < size; j++ {
			pGiven[i][j] = strconv.FormatFloat(round(p[i]), 'f', -1, 64)
		}
	}

	return pGiven
}

func entropy(p []float64) float64 {
	entropy := 0.0
	for _, px := range p {
		entropy -= px * math.Log2(px)
	}
	return round(entropy)
}

func jointEntropy(data [][]float64) float64 {
	entropy := 0.0
	for i := 0; i < len(data); i++ {
		for j := 0; j < len(data[i]); j++ {
			val := data[i][j]
			entropy -= val * math.Log2(val)
		}
	}
	return round(entropy)
}

func conditionalEntropy(pGiven [][]string, p []float64) float64 {
	entropy := 0.0
	for i := 0; i < len(p); i++ {
		sum := 0.0
		for j := 0; j < len(pGiven); j++ {
			val := fractionToFloat(pGiven[j][i])
			sum += val * math.Log2(val)
		}
		entropy += p[i] * -sum
	}

	return round(entropy)
}

func fractionToFloat(fraction string) float64 {
	parts := strings.Split(fraction, "/")
	if len(parts) < 2 {
		res, _ := strconv.ParseFloat(fraction, 64)
		return res
	}

	numerator, _ := strconv.ParseFloat(parts[0], 64)
	denominator, _ := strconv.ParseFloat(parts[1], 64)

	return numerator / denominator
}

func round(val float64) float64 { return math.Round(val*thousandForRounding) / thousandForRounding }
