import random


# retorna uma linha da matriz
def sudoku_line(n, matrix_sudoku):
    return matrix_sudoku[n]


# retorna uma coluna da matriz
def sudoku_column(n, matrix_sudoku):
    return [row[n] for row in matrix_sudoku]


# retorna um dos 9 grids da matriz
def sudoku_grid(n, matrix_sudoku):
    xi = int(n / 3) * 3
    yi = (n % 3) * 3
    xf = xi + 2
    yf = yi + 2
    grid = []
    for x in range(xi, xf + 1):
        for y in range(yi, yf + 1):
            grid.append(matrix_sudoku[x][y])
    return grid


# retorna um jogo de sudoku vazio. Utilizado para geração de filhos
def sudoku_empty():
    result = []
    for x in range(9):
        result.append([])
        for y in range(9):
            result[x].append(0)
    return result


# verifica se é valido -> sem duplicações
def sudoku_valid(sudoku_matrix):
    for n in range(9):
        if sudoku_duplicated(sudoku_line(n, sudoku_matrix)):
            return False
        if sudoku_duplicated(sudoku_column(n, sudoku_matrix)):
            return False
        if sudoku_duplicated(sudoku_grid(n, sudoku_matrix)):
            return False
    return True


# retorna a penalidade referente ao numero de casas erradas
def sudoku_penality(sudoku_list):
    p = 0
    for i in range(1, 10):
        if not sudoku_list.count(i):
            p += 1
    return p


def sudoku_duplicated(sudoku_list):
    duplicated = False
    for i in range(1, 10):
        if sudoku_list.count(i) > 1:
            duplicated = True
            break
    return duplicated


def sudoku_copy(sudoku_source):
    copy = sudoku_empty()
    for i in range(9):
        for j in range(9):
            copy[i][j] = sudoku_source[i][j]
    return copy

def sudoku_heuristic(sudoku_matrix):
    spots_filled = 0
    possible_digits_total = 0

    for i in range(9):
        for j in range(9):
            possible_digits = 0
            if sudoku_matrix[i][j] > 0:
                spots_filled += 1
            elif sudoku_matrix[i][j] == 0:
                temp_m = sudoku_copy(sudoku_matrix)
                for x in range(9):
                    temp_m[i][j] = x
                    if sudoku_valid(temp_m):
                        possible_digits += 1
                possible_digits_total += possible_digits/9
    return possible_digits_total - (81-spots_filled)

# função objetivo
def sudoku_goal(sudoku_matrix):
    p_L = 0
    p_C = 0
    p_G = 0
    # percorre todas as linhas,colunas e grid para obter a penalidade associada
    for x in range(9):
        p_L += sudoku_penality(sudoku_line(x, sudoku_matrix))
        p_C += sudoku_penality(sudoku_column(x, sudoku_matrix))
        p_G += sudoku_penality(sudoku_grid(x, sudoku_matrix))
    return (1 / (1 + p_L) + 1 / (1 + p_C) + 1 / (1 + p_G)) / 3


def generate_child(sudoku_source):
    if sudoku_valid(sudoku_source):
        empty_spots = []
        for i in range(9):
            for j in range(9):
                if sudoku_source[i][j] == 0:
                    empty_spots.append((i,j))

        #seleciona um spot vazio randomicamente
        random.seed(random.randint(1,100))
        spot = empty_spots[random.randint(1,len(empty_spots)-1)]
        child = sudoku_copy(sudoku_source)

        #tenta colocar randomicamente
        for n in range(1, 10):
            child[spot[0]][spot[1]] = n
            if sudoku_valid(child):
                fitness = sudoku_heuristic(child)
                return child, fitness, len(empty_spots)-1

        #tenta colocar sequenciamente
        child = sudoku_copy(sudoku_source)
        for s in empty_spots:
            for n in range(1,10):
                child[s[0]][s[1]] = n
                if sudoku_valid(child):
                    fitness = sudoku_heuristic(child)
                    return child, fitness, len(empty_spots) - 1
                child[s[0]][s[1]] = 0
        print('Nao encontrou filho possivel')


# gera os K filhos a partir de um source. Retorna um array com tuplas(sudoku_game,cost)
def generate_children(k, sudoku_source):
    children = []
    for x in range(k):
        child = generate_child(sudoku_source)
        if child is not None:
            children.append(child)
    return children


def sudoku_shuffle(child, sudoku_inicial, sudoku_source):
    final = [
        [8, 3, 7, 1, 9, 4, 6, 2, 5],
        [5, 4, 9, 6, 2, 3, 7, 8, 1],
        [6, 2, 1, 7, 8, 5, 9, 3, 4],
        [2, 5, 6, 8, 1, 7, 4, 9, 3],
        [4, 1, 3, 5, 6, 9, 2, 7, 8],
        [9, 7, 8, 3, 4, 2, 5, 1, 6],
        [1, 6, 4, 2, 7, 8, 3, 5, 9],
        [7, 9, 5, 4, 3, 1, 8, 6, 2],
        [3, 8, 2, 9, 5, 6, 1, 4, 7]]
    return final


# ordena a lista de fihos e escolhe os K melhores
def sudoku_selection(k, children):
    sorted_children = sorted(children, key=lambda child: child[1], reverse=True)
    return sorted_children[:k]


# funcao checagem
def sudoku_check(children):
    for child in children:
        if child[1] >= 1:
            return child


def sudoku_check_element(element, line, col, sudoku_matrix):
    valid = True
    # checa linha
    if sudoku_line(line, sudoku_matrix).count(element) > 1:
        valid = False

    # checa coluna
    if sudoku_column(col, sudoku_matrix).count(element) > 1:
        valid = False
    return valid


def algorithm():
    # posicao inicial
    initial = [
        [8, 3, 0, 1, 0, 0, 6, 0, 5],
        [0, 0, 0, 0, 0, 0, 0, 8, 0],
        [0, 0, 0, 7, 0, 0, 9, 0, 0],
        [0, 5, 0, 0, 1, 7, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 3, 4, 0, 0, 1, 0],
        [0, 0, 4, 0, 0, 8, 0, 0, 0],
        [0, 9, 0, 0, 0, 0, 0, 0, 0],
        [3, 0, 2, 0, 0, 6, 0, 4, 7]]

    # tamanho do feixe
    k = 5

    # gerando a populacao inicial
    pop = generate_children(k, initial)

    aux = 0
    found_solution = False
    while not found_solution:
        print('Geracao: ' + str(aux))
        # gerando os filhos
        children = []
        for i in pop:
            children += generate_children(k, i[0])
        for child in children:
            if sudoku_goal(child[0]) >= 1:
                found_solution = True
                print(child)
                break
        pop += children
        pop = sudoku_selection(k, pop)

        max = pop[0][1]
        min = pop[k - 1][1]
        avg = 0
        for p in pop:
            avg += p[1]
        avg = avg / k

        print('MAX: ' + str(max))
        print('MIN: ' + str(min))
        print('AVG: ' + str(avg))
        print('SPOTS:'+ str(81-pop[0][2]))
        print('-----------------')
        aux += 1


def test():
    # TESTE DE FUNCOES AUXILIARES
    print('\nFUNCOES AUXILIARES')
    a = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9],
        [1, 2, 3, 4, 5, 6, 7, 8, 9]]
    initial = [
        [8, 3, 0, 1, 0, 0, 6, 0, 5],
        [0, 0, 0, 0, 0, 0, 0, 8, 0],
        [0, 0, 0, 7, 0, 0, 9, 0, 0],
        [0, 5, 0, 0, 1, 7, 0, 0, 0],
        [0, 0, 3, 0, 0, 0, 2, 0, 0],
        [0, 0, 0, 3, 4, 0, 0, 1, 0],
        [0, 0, 4, 0, 0, 8, 0, 0, 0],
        [0, 9, 0, 0, 0, 0, 0, 0, 0],
        [3, 0, 2, 0, 0, 6, 0, 4, 7]]
    final = [
        [8, 3, 7, 1, 9, 4, 6, 2, 5],
        [5, 4, 9, 6, 2, 3, 7, 8, 1],
        [6, 2, 1, 7, 8, 5, 9, 3, 4],
        [2, 5, 6, 8, 1, 7, 4, 9, 3],
        [4, 1, 3, 5, 6, 9, 2, 7, 8],
        [9, 7, 8, 3, 4, 2, 5, 1, 6],
        [1, 6, 4, 2, 7, 8, 3, 5, 9],
        [7, 9, 5, 4, 3, 1, 8, 6, 2],
        [3, 8, 2, 9, 5, 6, 1, 4, 7]]
    print(sudoku_line(0, a))
    print(sudoku_column(0, a))
    print(sudoku_grid(8, a))
    print(sudoku_empty())

    # TESTE DA FUNCAO PENALIDADE
    print('\nFUNCAO PENALIDADE')
    print(sudoku_penality(sudoku_grid(0, initial)))
    print(sudoku_penality(sudoku_line(0, final)))

    # TESTE DA FUNCAO OBJETIVO
    print('\nFUNCAO OBJETIVO')
    print(sudoku_goal(initial))
    print(sudoku_goal(final))

    # TESTE DA FUNCAO SHUFFLE
    print('\nFUNCAO SHUFFLE')
    shuffled = sudoku_shuffle(initial, initial)
    print(shuffled)
    penality_line = 0
    for x in shuffled:
        penality_line += sudoku_penality(x)
    print('Penalidade linha: ' + str(penality_line))

    # TESTE DA FUNCAO GENERATE_CHILDREN
    print('\nFUNCAO GENERATE CHILDREN')
    k = 3
    children = generate_children(k, initial, initial)
    print(children)

    # TESTE DA FUNCAO SELECTION
    print('\nFUNCAO SELECTION')
    n = 1
    print(sudoku_selection(n, children))

    # TESTE DA FUNCAO CHECK ELEMENT
    print('\nFUNCAO CHECK ELEMENT')
    for l in range(8):
        for c in range(8):
            print(sudoku_check_element(children[0][0][l][c], l, c, children[0][0]))


def main():
    algorithm()


if __name__ == "__main__":
    main()
