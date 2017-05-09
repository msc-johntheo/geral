import random

# definindo os elementos
digits = '123456789'
rows = 'ABCDEFGHI'
cols = digits


def cross(A, B):
    "Funcao auxiliar para realizar o produto cruzado dos elementos de A e B"
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


def grid_hash(grid):
    """Converte a grid em um dicionario de {square: valor} com '0' or '.' para os vazios."""
    chars = [c for c in grid if c in digits or c in '0.']
    assert len(chars) == 81
    return dict(zip(squares, chars))


def empty_squares(grid_dict):
    return {k: v for k, v in grid_dict.items() if v in '0.'}
