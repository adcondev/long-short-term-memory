def FPGA_Weights(forget, input, candidate, output):
    for gateVals, gateNames in zip([forget, input, candidate, output], ['F', 'I', 'C', 'O']):
        with open("Gates" + "gateNames" + ".coe", "w") as file:
            for val in gateVals.reshape(1, -1):
                file.write(str(bin(np.float16(val).view('H'))[2:].zfill(16)) + ",")