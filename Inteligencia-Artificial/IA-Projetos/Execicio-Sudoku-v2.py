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


# retorna uma lista com s spots vazios
def sudoku_empty_spots(sudoku_matrix):
    empty_spots = []
    for i in range(9):
        for j in range(9):
            if sudoku_matrix[i][j] == 0:
                empty_spots.append((i, j))
    return empty_spots


# retorna um jogo de sudoku vazio. Utilizado para geração de filhos
def sudoku_empty():
    result = []
    for x in range(9):
        result.append([])
        for y in range(9):
            result[x].append(0)
    return result


# copia uma matriz sudoku
def sudoku_copy(sudoku_matrix):
    copy = sudoku_empty()
    for i in range(9):
        for j in range(9):
            copy[i][j] = sudoku_matrix[i][j]
    return copy


# verificar duplições em uma lista(linha,coluna,grid)
def sudoku_duplicated(sudoku_list):
    duplicated = False
    for i in range(1, 10):
        if sudoku_list.count(i) > 1:
            duplicated = True
            break
    return duplicated


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


# heuristica para cálculo de fitness
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
                possible_digits_total += possible_digits / 9
    #return possible_digits_total - (81 - spots_filled)
    return 81 - spots_filled - possible_digits_total
    #return possible_digits_total

#embaralha uma lista
def list_shuffle(orig):
    dest = orig[:]
    random.shuffle(dest)
    return dest

# gera os filhos do estado atual somente com estados válidos
def generate_children(k, sudoku_matrix):
    empty_spots = sudoku_empty_spots(sudoku_matrix)
    filled_spots = (81 - len(empty_spots))
    nums = [1,2,3,4,5,6,7,8,9]

    empty_spots = list_shuffle(empty_spots)
    nums = list_shuffle(nums)
    children = []
    for s in empty_spots:
        child = sudoku_copy(sudoku_matrix)
        for n in nums:
            child[s[0]][s[1]] = n
            if sudoku_valid(child):
                fitness = sudoku_heuristic(child)
                children.append((child, fitness, filled_spots))
                if len(children) == k:
                    return children
            child = sudoku_copy(sudoku_matrix)
    return children


# Seleciona os k melhores membros da população
def select(k,population):
    sorted_population = sorted(population, key=lambda pop: pop[1], reverse=False)
    return sorted_population[:k]


# Funcao objetivo que verifica se todos os spots estao preenchidos
def goal(population):
    for p in population:
        if p[2] == 81: #todas os spots preenchidos
            return p
    return None


# Imprime indivíduo
def print_individual(sudoku_matrix):
    for line in range(9):
        print_line = ''
        if line == 3 or line == 6:
            print('---------------------')
        for col in range(9):
            if col == 2 or col == 5:
                print_line += str(sudoku_matrix[line][col]) + ' | '
            else:
                print_line += str(sudoku_matrix[line][col]) + ' '
        print(print_line)


# Retorna informações da populacao(max,min,avg,filled)
def info_population(population):
    avg = 0
    for p in population:
        avg += p[1]
    avg = avg/len(population)
    return population[0][1], population[len(population) - 1][1], avg, population[0][2]


# Imprime info da populacao
def print_info_population(population):
    info = info_population(population)
    print('MAX: ' + str(info[0]))
    print('MIN: ' + str(info[1]))
    print('AVG: ' + str(info[2]))
    print('SPOTS: ' + str(info[3]))

def beam_search():
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
    k = 70

    # populacao inicial
    population = generate_children(k, initial)

    #verifica se já alcancou o objetivo
    solution = goal(population)
    if solution is not None:
        return solution

    generation = 0
    while True:
        print('Geracao: ' + str(generation))
        print_info_population(population)
        children = []
        for p in population:
            children += generate_children(k, p[0])

        #condicional de maximo local
        if len(children) == 0:
            print('MAXIMO LOCAL - MELHOR SOLUCAO')
            best_solution = population[0][0]
            print_individual(best_solution)
            return best_solution


        # verifica se já alcançou o objetivo
        solution = goal(children)
        if solution is not None:
            print('SOLUCAO')
            print_individual(solution[0][0])
            return solution
        else:
            print('SELECIONADO')
            selected = select(k, children)
            print_individual(selected[0][0])
            population = selected
        generation += 1
        print('============================================')


def test():
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

    print(sudoku_heuristic(final))


def main():
    beam_search()


if __name__ == "__main__":
    main()