import ctypes
class ball_k_means:
    def __init__(self, isRing = 0):
        self.isRing = isRing

    def fit(self, s1, s2):
        # isRing == 0 represent alg with rings
        # isRing == others represent alg with no rings
        if self.isRing == 0:
            print("have ring: ")
            dll = ctypes.cdll.LoadLibrary('./Ring.dll')
            dll.ball_kmeans.argtypes = (ctypes.c_char_p, ctypes.c_char_p)
            dll.ball_kmeans(s1.encode('utf-8'), s2.encode('utf-8'))
        else:
            print("have no ring: ")
            dll = ctypes.cdll.LoadLibrary('./noRing.dll')
            dll.ball_kmeans.argtypes = (ctypes.c_char_p, ctypes.c_char_p)
            dll.ball_kmeans(s1.encode('utf-8'), s2.encode('utf-8'))

if __name__ == '__main__':
    dataset_address = "./data/Dataset2.txt"
    centriod_address = ""
    clf = ball_k_means(isRing=0)
    print(clf)
    clf.fit(dataset_address, centriod_address)
