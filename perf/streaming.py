import numpy as np
import npstreams as nps


def describe_streaming_functions():

    def describe_vectorized():

        def _vectorized():
            test_array = np.random.rand(100, 100, 100)

            result = test_array.mean(axis=-1).mean(axis=-1)

        def its_fast(benchmark):
            benchmark(_vectorized)

    def describe_loops():

        def _loop():
            test_array = np.random.rand(100, 100, 100)

            result = np.empty(100)
            for i, x in enumerate(np.random.rand(100)):
                sum_ys = 0
                for y in np.random.rand(100):
                    sum_zs = 0
                    for z in np.random.rand(100):
                        sum_zs += z/100
                    sum_ys += sum_zs + y/100

                result[i] = sum_ys/100

        def its_slow(benchmark):
            benchmark(_loop)

    def describe_streams():

        def _stream():
            test_array = np.random.rand(100, 100, 100)

            stream = (ys for yarr in test_array for ys in yarr)

            y_sums = nps.imean(stream)

            result = nps.mean(y_sums)

        def its_hopefully_fast(benchmark):
            benchmark(_stream)
