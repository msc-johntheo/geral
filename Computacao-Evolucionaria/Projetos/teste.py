def decode(individual, i_size=IND_SIZE, g_size=GEN_SIZE):
    """Recebe um individuo codificado e decodifica."""
    result = []
    for a in range(0, i_size, g_size):
        print(a)
        print(i_size)
        print(g_size)
        gene = ''.join(str(individual[i]) for i in range(a, a + GEN_SIZE))
        index_rota = int(gene, 2)
        rota = '' if index_rota > i_size - 1 else rotas[index_rota]
        result.append(rota)
    return result


def main():
    a = [0, 0, 0, 1]
    print(decode(a, len(a), 2))


if __name__ == "__main__":
    main()
