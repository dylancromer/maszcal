package main

import ("fmt";
        "gonum.org/v1/gonum/mat";
)

func trapz(xs mat.VecDense, ys mat.VecDense) float64 {

}

func main() {
    y_data := []float64{1,2,3,4}
    x_data := []float64{1,1,1,1}


    xs := mat.NewVecDense(4, y_data)
    ys := mat.NewVecDense(4, x_data)
    var integral float64

    trapz(xs, ys)
    fmt.Println("a_matrix + b_matrix")
    fmt.Println(c_matrix)
}
