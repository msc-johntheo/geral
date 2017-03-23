from scipy.spatial import distance
import time
MANHATTAN = 1
PIECES = 2

class State:
    def __init__(self, configuration, actual_cost, heuristic_cost, came_from):
        self.configuration = configuration
        self.actual_cost = actual_cost
        self.heuristic_cost = heuristic_cost
        self.total_cost = self.actual_cost + self.heuristic_cost
        self.came_from = came_from
    def __cmp__(self,other):
        return (self.total_cost > other.total_cost) - (self.total_cost < other.total_cost)

# funcao auxiliar para retornar posicao no jogo
def index2d(index):
    return int(index / 3), index % 3

#calcula o somatorio da distancia manhatan de todas as peças(incluido a casa vazia)
def manhattan(matrix1, matrix2):
    dist = 0
    for i in matrix1:
        dist += distance.cityblock(index2d(matrix1.index(i)),index2d(matrix2.index(i)))
    return dist

#calcula o somatorio de peças fora do ligar(incluido a casa vazia)
def pieces(matrix1,matrix2):
    pieces = 0
    for i in matrix1:
        if matrix1.index(i) != matrix2.index(i):
            pieces += 1
    return pieces

def heuristic(matrix1, matrix2, heuristic):
    if heuristic == MANHATTAN:
        return manhattan(matrix1, matrix2)
    elif heuristic == PIECES:
        return pieces(matrix1, matrix2)

def generate_children(state,final_configuration,method):
    # lista com os filhos
    children = []

    # encontrando a posicao da casa vazia e as casas vizinhas
    empty_pos = state.configuration.index(0)
    above_pos = empty_pos - 3
    below_pos = empty_pos + 3
    right_pos = empty_pos + 1
    left_pos = empty_pos - 1

    # trocando com a posicao ACIMA caso existir
    if above_pos >= 0:
        new_conf = state.configuration.copy()
        new_conf[empty_pos] = new_conf[above_pos]
        new_conf[above_pos] = 0
        child = State(new_conf, state.actual_cost + 1, heuristic(new_conf, final_configuration, method), state)
        children.append(child)

    # trocando com a posicao ABAIXO caso existir
    if below_pos < len(state.configuration):
        new_conf = state.configuration.copy()
        new_conf[empty_pos] = new_conf[below_pos]
        new_conf[below_pos] = 0
        child = State(new_conf, state.actual_cost + 1, heuristic(new_conf, final_configuration, method), state)
        children.append(child)

    # trocando com a posicao DIREITA
    if right_pos < len(state.configuration):
        new_conf = state.configuration.copy()
        new_conf[empty_pos] = new_conf[right_pos]
        new_conf[right_pos] = 0
        child = State(new_conf, state.actual_cost + 1, heuristic(new_conf, final_configuration, method), state)
        children.append(child)

    # trocando com a posicao ESQUERDA caso existir
    if left_pos >= 0:
        new_conf = state.configuration.copy()
        new_conf[empty_pos] = new_conf[left_pos]
        new_conf[left_pos] = 0
        child = State(new_conf, state.actual_cost + 1, heuristic(new_conf, final_configuration, method), state)
        children.append(child)

    return children


def add_to_border(state_list, border):
    already_on_border = False
    for state in state_list:
        for border_state in border:
            if state.configuration == border_state.configuration:
                already_on_border = True
                if state.actual_cost < border_state.actual_cost:
                    border[border.index(border_state)] = state
        if not already_on_border:
            border.append(state)


def print_as_table(configuration):
    l1=""
    l2=""
    l3=""
    for i in configuration:
       if configuration.index(i) < 3:
           l1 += str(i)
       elif 3 <= configuration.index(i) < 6:
           l2 += str(i)
       else:
           l3 += str(i)
    print(l1)
    print(l2)
    print(l3)
    print("---")


def algorithm(initial_state, final_state, heuristic_method):
    start_time = time.time()
    generation = 0
    method = heuristic_method

    # fronteira
    border = []

    # estados visitados
    visited = []

    # representando os estados
    si = initial_state
    sf = final_state

    found_solution = False

    # Criando o estado inicial com todos os valores
    start_state = State(si, 0, heuristic(si, sf, method), None)

    # adicionando o estado inicial à fronteira
    border.append(start_state)

    # enquanto a fronteira nao estiver vazia
    while len(border) > 0:
        generation += 1

        # ordenando para poder obter o melhor a cada iteração
        border.sort(key=lambda state: state.total_cost)

        # selecionando o melhor e colocando na lista dos nós visitados
        actual_state = border[0]
        visited.append(actual_state)

        # removendo o nó visitado
        del (border[0])

        # verificando se o estado já é o final
        if actual_state.configuration == sf:
            found_solution = True
            break
        else:
            # gerando os estados filhos e adicionado à fronteira
            add_to_border(generate_children(actual_state, sf, method), border)

    result = []

    if found_solution:
        state = visited.pop()

        while state is not start_state:
            result.append(state)
            state = state.came_from

        #incuindo o no inicial ao resultado
        result.append(start_state)
        #inverntendo a ordem para imprimir o passo correto
        result.reverse()

        print('PASSOS PARA A SOLUCAO' + '\n')
        for r in result:
            print_as_table(r.configuration)
    else:
        print('Nao encontrou solucao')
    end_time = time.time()
    print('Tempo total: ' + str(end_time - start_time) +str(' segundos'))
    print('Geracoes: ' + str(generation))
    print('Profundidade: ' + str(len(result)))


def main():
    initial_state = [7, 2, 4,
                     5, 0, 6,
                     8, 3, 1]

    final_state = [0, 1, 2,
                   3, 4, 5,
                   6, 7, 8]

    heuristic_method = MANHATTAN

    algorithm(initial_state, final_state, heuristic_method)


if __name__ == "__main__":
    main()
