#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;
namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    printf("begin\n");
    //float *Z=new float [batch*k];
    auto Z = std::vector<std::vector<float>>(batch, std::vector<float>(k, 0.0));
    //auto Z =std::vector<float>(batch*k);
    for(size_t i=0;i<m;i+=batch){
        if(i+batch>m)batch=m-i;
        const float *X_batch=X+i*n;
        const unsigned char *y_batch=y+i;
        //求Z~batch*k，并归一化
        for(size_t line=0;line<batch;line++){
            float sum_line=0;
            //printf("%d ++++++++++++line-\n",line);
            for(size_t row=0;row<k;row++){
                float tmp=0;Z[line][row] = 0.0;
                for(size_t j=0;j<n;j++){
                    tmp+=X[(line+i)*n+j]*theta[j*k+row];
                }
                tmp= exp(tmp);
                //Z[line*batch+row]=tmp;
                Z[line][row]=tmp;
                sum_line+=tmp;
            }
            //归一化
            for(size_t row=0;row<k;row++){
                Z[line][row]/=sum_line;
            }

        }
        //printf("%d--end Z\n",i);
        //
        for(size_t line=0;line<batch;line++){
            //Z[line*batch+y_batch[line]]-=1;
            Z[line][y[line+i]]-=1.0;
        }

        //delta
        for(size_t line=0;line<n;line++){
            for(size_t row=0;row<k;row++){
                float tmp=0;
                for(size_t j=0;j<batch;j++){
                    tmp+=X[(j+i)*n+line]*Z[j][row];
                }
                theta[line*k+row]-=lr*tmp/batch;
            }
        }

    }

    //delete[] Z;
    printf("end");
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */

PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
