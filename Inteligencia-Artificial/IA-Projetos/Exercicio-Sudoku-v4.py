# definindo os elementos
digits = '123456789'
rows = 'ABCDEFGHI'
cols = digits


def cross(A, B):
    "Produto cruzado dos elementos de A e B"
    return [a + b for a in A for b in B]


# cada square é um quadrado da matriz 9x9 do sudoku.
squares = cross(rows, cols)

# cada square tem/pertence 9 units(linha, coluna, grid/. Unitlist é a lista de todas as colunas, linhas e grids
unitlist = ([cross(rows, c) for c in cols] +
            [cross(r, cols) for r in rows] +
            [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123', '456', '789')])

# dicionario para retornar as units de um determinado square
units = dict((s, [u for u in unitlist if s in u]) for s in squares)

# peers de um square S sao todos os squares das 3 unidades relacionadas a S exceto o próprio S
peers = dict((s, set(sum(units[s], [])) - set([s])) for s in squares)


def grid_values(grid):
    """Converte a grid em um dicionario de {square: valor} com '0' or '.' para os vazios."""
    chars = [c for c in grid if c in digits or c in '0.']
    assert len(chars) == 81
    return dict(zip(squares, chars))


##CONSTRAINT PROPAGATION

def eliminate(values, s, d):
    """Eliminate d from values[s]; propagate when values or places <= 2.
    Return values, except return False if a contradiction is detected."""
    if d not in values[s]:
        return values  ## Already eliminated
    values[s] = values[s].replace(d, '')
    ## (1) If a square s is reduced to one value d2, then eliminate d2 from the peers.
    if len(values[s]) == 0:
        return False  ## Contradiction: removed last value
    elif len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, s2, d2) for s2 in peers[s]):
            return False
    ## (2) If a unit u is reduced to only one place for a value d, then put it there.
    for u in units[s]:
        dplaces = [s for s in u if d in values[s]]
        if len(dplaces) == 0:
            return False  ## Contradiction: no place for this value
        elif len(dplaces) == 1:
            # d can only be in one place in unit; assign it there
            if not assign(values, dplaces[0], d):
                return False
    return values


def assign(values, s, d):
    """Eliminate all the other values (except d) from values[s] and propagate.
    Return values, except return False if a contradiction is detected."""
    other_values = values[s].replace(d, '')
    if all(eliminate(values, s, d2) for d2 in other_values):
        return values
    else:
        return False


def parse_values(grid_v):
    """Converte um dict de valores de uma grid em um dict de valores possíveis usando contraint propagation
    O dict fica da seguinte forma {square:possible_digits}"""

    # Para início, todos os quadrados podem ter qualquer dígito
    values = dict((s, digits) for s in squares)
    for k, v in grid_v.items():
        if len(v) > 1:
            grid_v[k] = '.'

    for s, d in grid_v.items():
        if d in digits and not assign(values, s, d):
            return False  ## (Fail se encontrar uma contradição)
    return values


### MÉTODOS AUXILIARES DO ALGORTIMO DE BEAM SEARCH
def sort_grid(grid_values):
    """Retorna um dict com os valores ordenado em ordem crescente de possibilidade"""
    return dict((k, v) for k, v in (sorted(grid_values.items(), key=lambda x: len(x[1]))) if len(v) > 1)


def grid_fitnesss(sorted_values):
    """Calcula o fitness de uma grid_values(dict) contando o numero de possibilidades para os squares ainda nao preenchidos"""
    fitness = 0
    for k, v in sorted_values.items():
        if len(v) > 1:
            fitness += len(v)
    return fitness


def generate_children(individual, k):
    """gera os k filhos de um indivíduo"""
    children = []
    possibles = sort_grid(individual)
    for key, values in possibles.items():
        for v in values:
            child = individual.copy()
            child[key] = v
            child = parse_values(child)
            if child:
                children.append(child)
                k -= 1

            if k == 0:
                return children

    return children


def display(values):
    "Mostra o dict de valores no formado grid 2-D"
    width = 1 + max(len(values[s]) for s in squares)
    line = '+'.join(['-' * (width * 3)] * 3)
    result = ''
    for r in rows:
        result += (''.join(values[r + c].center(width) + ('|' if c in '36' else '') for c in cols)) + '\n'
        if r in 'CF':
            result += line + '\n'
    result += '\n'
    print(result)
    return result


def beam_search(state, k):
    """Exuecuta o algoritmo de busca em feixe, Recebe um grid inicial e um k de filhos a serem gerados por feixe"""
    initial_values = parse_values(grid_values(state))
    if not initial_values:
        print('INVALIDO')
        return
    if grid_fitnesss(initial_values) == 0:
        display(initial_values)
        return
    else:
        population = [initial_values]
        children = []
        generation = 1
        while len(population):
            print('Generation: {}'.format(generation))
            for p in population:
                children += generate_children(p, k)
                children = sorted(children, key=lambda child: grid_fitnesss(child))
            if grid_fitnesss(children[0]) == 0:
                display(children[0])
                return
            else:
                population = children[:k]
                display(children[0])
                generation += 1
                print('--------------------')


def main():
    initial = """
    8 3 . |1 . . |6 . 5
    . . . |. . . |. 8 . 
    . . . |7 . . |9 . . 
    ------+------+------
    . 5 . |. 1 7 |. . . 
    . . 3 |. . . |2 . . 
    . . . |3 4 . |. 1 . 
    ------+------+------
    . . 4 |. . 8 |. . . 
    . 9 . |. . . |. . . 
    3 . 2 |. . 6 |. 4 7 
    """
    initial2 = """
    8 3 . |1 . . |6 . 5
    . . . |. . . |. . . 
    . . . |. . . |. . . 
    ------+------+------
    . . . |. . 7 |. . . 
    . . . |. . .|. . . 
    . . . |. 4 . |. 1 . 
    ------+------+------
    . . . |. . 8 |. . . 
    . . . |. . . |. . . 
    3 . . |. . 6 |. 4 7 
        """
    k = 3

    beam_search(initial2, k)


if __name__ == "__main__":
    main()
