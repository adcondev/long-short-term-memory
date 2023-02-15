class Lista:
    def __init__(self):
        self.list = []
    def imprimirLista(self):
        print(self.list)
    def add(self, n):
        self.list.append(n)
    def remove(self):
        return self.list.pop()
miLista = Lista()
miLista2 = Lista()
miLista2.add(10)
miLista.add(5)
ultimoValor = miLista.remove()
miLista.imprimirLista()
miLista2.imprimirLista()
print(ultimoValor)
