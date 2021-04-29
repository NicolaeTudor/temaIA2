from __future__ import annotations

import copy
from dataclasses import dataclass, field
from tkinter import *
from typing import Any, List, Tuple

from PIL import ImageTk, Image
import numpy as np

import custom_heap

SIZE_OF_BOARD = 768
NUMBER_OF_CELLS = 5
NUMBER_OF_WALLS = 4
NUMBER_OF_LINES = NUMBER_OF_CELLS + 1
PLAYER_1_COLOR = '#0492CF'
PLAYER_2_COLOR = '#EE4035'
WALL_COLOR = '#1cce96'
MOVE_COLOR = '#ecf394'
BACKGROUND_COLOR = '#2b2b2b'
DISTANCE_BETWEEN_LINES = SIZE_OF_BOARD / NUMBER_OF_LINES
EDGE_WIDTH = 0.2 * DISTANCE_BETWEEN_LINES
OFFSET = DISTANCE_BETWEEN_LINES / 2
PIECE_OFFSET = OFFSET + EDGE_WIDTH / 2
GRID_SIZE = NUMBER_OF_LINES - 2 + NUMBER_OF_CELLS
CENTER_CELL_OFFSET = DISTANCE_BETWEEN_LINES * (NUMBER_OF_CELLS // 2)
CELL = 1
LINE = 0
INTERSECTION = -1
WALL = -2
PLAYER_1 = -3
PLAYER_2 = -4
MARKED_MOVE = -5
PLAYER_1_MARK = 'PLAYER_1'
PLAYER_2_MARK = 'PLAYER_2'
DEPTH_MAX = 4
MAX_SCORE = 100


class QuoridorBoard:
    def __init__(self):
        self.state = None
        self.window = Tk()
        self.window.title('Quoridor - Nicolae Tudor-Iulian')
        self.window.resizable(0, 0)
        self.canvas = Canvas(self.window, width=SIZE_OF_BOARD, height=SIZE_OF_BOARD, background=BACKGROUND_COLOR)
        self.canvas.pack()

        self.player_1_icon = ImageTk.PhotoImage(Image.open("Images/piece_sprite.png")
                                                .crop((0, 0, 310, 310))
                                                .resize((int(DISTANCE_BETWEEN_LINES - EDGE_WIDTH),
                                                         int(DISTANCE_BETWEEN_LINES - EDGE_WIDTH))
                                                        )
                                                )
        self.player_2_icon = ImageTk.PhotoImage(Image.open("Images/piece_sprite.png")
                                                .crop((310, 0, 620, 310))
                                                .resize((int(DISTANCE_BETWEEN_LINES - EDGE_WIDTH),
                                                         int(DISTANCE_BETWEEN_LINES - EDGE_WIDTH)
                                                         ))
                                                )
        self.player_1 = None
        self.player_2 = None
        self.player_move = False
        self.first_move = True
        self.init_game()

    def mainloop(self):
        self.window.mainloop()

    def refresh_board(self):
        for i in range(NUMBER_OF_LINES):
            x = i * DISTANCE_BETWEEN_LINES + OFFSET
            self.canvas.create_line(x, OFFSET,
                                    x, SIZE_OF_BOARD - OFFSET,
                                    fill='gray', dash=(2, 2))
            self.canvas.create_line(OFFSET, x,
                                    SIZE_OF_BOARD - OFFSET, x,
                                    fill='gray', dash=(2, 2))

        self.canvas.create_line(OFFSET, OFFSET,
                                SIZE_OF_BOARD - OFFSET, OFFSET,
                                SIZE_OF_BOARD - OFFSET, SIZE_OF_BOARD - OFFSET,
                                OFFSET, SIZE_OF_BOARD - OFFSET,
                                OFFSET, OFFSET,
                                fill=WALL_COLOR, width=EDGE_WIDTH, tag='wall'
                                )

        self.player_1 = self.canvas.create_image(CENTER_CELL_OFFSET + PIECE_OFFSET,
                                                 DISTANCE_BETWEEN_LINES * (NUMBER_OF_CELLS - 1) + PIECE_OFFSET,
                                                 anchor=NW, image=self.player_1_icon, tag="player_1")

        self.player_2 = self.canvas.create_image(CENTER_CELL_OFFSET + PIECE_OFFSET,
                                                 PIECE_OFFSET,
                                                 anchor=NW, image=self.player_2_icon, tag="player_2")

        text_wall_player_1 = f'Player 1 walls: {NUMBER_OF_WALLS}'
        text_wall_player_2 = f'Player 2 walls: {NUMBER_OF_WALLS}'
        self.canvas.create_text(OFFSET + 5 * len(text_wall_player_1),
                                OFFSET / 2,
                                font="cmr 15 bold", text=text_wall_player_1, fill=PLAYER_1_COLOR,
                                tag='player_1_wall_count')

        self.canvas.create_text(SIZE_OF_BOARD - OFFSET - 5 * len(text_wall_player_2),
                                OFFSET / 2,
                                font="cmr 15 bold", text=text_wall_player_2, fill=PLAYER_2_COLOR,
                                tag='player_2_wall_count')

        text_turn = 'Turn: Player 1'

        self.canvas.create_text(OFFSET + 5 * len(text_turn),
                                SIZE_OF_BOARD - OFFSET / 2,
                                font="cmr 15 bold", text=text_turn, fill=PLAYER_1_COLOR,
                                tag='player_turn')

    def init_game(self):
        self.canvas.bind('<Button-1>', self.click)

        board = np.empty(shape=(GRID_SIZE, GRID_SIZE))
        self.refresh_board()
        is_row = False
        for i in range(GRID_SIZE):
            is_col = False
            for j in range(GRID_SIZE):
                if is_row and is_col:
                    board[i][j] = INTERSECTION  # intersection
                elif is_row or is_col:
                    board[i][j] = LINE
                else:
                    board[i][j] = CELL
                is_col = not is_col
            is_row = not is_row

        player_1_position = (GRID_SIZE - 1, GRID_SIZE // 2)
        player_2_position = (0, GRID_SIZE // 2)
        player_1_walls = NUMBER_OF_WALLS
        player_2_walls = NUMBER_OF_WALLS
        self.state = State(board, PLAYER_1_MARK, DEPTH_MAX,
                           player_1_position, player_2_position,
                           player_1_walls, player_2_walls)

    @staticmethod
    def is_final(state: State):
        if state.turn == PLAYER_1_MARK and state.player_1_position[0] == 0:
            return True
        elif state.turn == PLAYER_2_MARK and state.player_2_position[0] == GRID_SIZE - 1:
            return True

        return False

    @staticmethod
    def swap_turn(state: State):
        if state.turn == PLAYER_1_MARK:
            state.turn = PLAYER_2_MARK
        else:
            state.turn = PLAYER_1_MARK

    @staticmethod
    def is_in_board(i, j):
        if i == -1 or i == GRID_SIZE or j == -1 or j == GRID_SIZE:
            return False
        return True

    @staticmethod
    def convert_to_cell_coordinate(abs_coordinate):
        cell_coord = int((abs_coordinate - OFFSET) // DISTANCE_BETWEEN_LINES) * 2
        if cell_coord >= GRID_SIZE or cell_coord < 0:
            cell_coord = -1
        return cell_coord

    @staticmethod
    def convert_to_cell_position(abs_position):
        cell_position = tuple(map(QuoridorBoard.convert_to_cell_coordinate, reversed(abs_position)))
        return cell_position

    @staticmethod
    def convert_to_grid_coordinate(abs_coordinate):
        grid_coord = int((abs_coordinate - OFFSET - DISTANCE_BETWEEN_LINES / 4) // (DISTANCE_BETWEEN_LINES / 2))
        if grid_coord >= GRID_SIZE or grid_coord < 0:
            grid_coord = -1
        return grid_coord

    @staticmethod
    def convert_to_grid_position(abs_position):
        grid_position = tuple(map(QuoridorBoard.convert_to_grid_coordinate, reversed(abs_position)))
        return grid_position

    @staticmethod
    def convert_to_line_position(grid_position):
        logical_position = (None, -1, -1)
        # They have the same parity
        if (grid_position[0] & 1) != (grid_position[1] & 1):
            if grid_position[0] & 1 == 0:  # We are on a col
                logical_position = ('col', grid_position[1] // 2 + 1, (grid_position[0]) // 2)
            else:
                logical_position = ('row', grid_position[0] // 2 + 1, (grid_position[1]) // 2)

        return logical_position

    @staticmethod
    def undo_edge(state: State, grid_position):
        board = state.board
        i, j = grid_position
        logical_position = QuoridorBoard.convert_to_line_position(grid_position)
        if logical_position[0] == 'row':
            for add in range(3):
                if (add & 1) == 0:
                    board[i][j + add] = LINE
                else:
                    board[i][j + add] = INTERSECTION
        if logical_position[0] == 'col':
            for add in range(3):
                if (add & 1) == 0:
                    board[i + add][j] = LINE
                else:
                    board[i + add][j] = INTERSECTION

    @staticmethod
    def try_edge(state: State, grid_position):
        board = state.board
        i, j = grid_position
        if i == -1 or j == -1:
            return False
        logical_position = QuoridorBoard.convert_to_line_position(grid_position)
        if logical_position[0] is None:
            return False

        if logical_position[0] == 'row':
            for add in range(3):
                if not QuoridorBoard.is_in_board(i, j + add) or board[i][j + add] == WALL:
                    return False
            for add in range(3):
                board[i][j + add] = WALL
        if logical_position[0] == 'col':
            for add in range(3):
                if not QuoridorBoard.is_in_board(i + add, j) or board[i + add][j] == WALL:
                    return False
            for add in range(3):
                board[i + add][j] = WALL

        can_end = True
        if (QuoridorBoard.shortest_path(state, PLAYER_1_MARK) == -1 or
                QuoridorBoard.shortest_path(state, PLAYER_2_MARK) == -1):
            can_end = False

        QuoridorBoard.undo_edge(state, grid_position)

        return can_end

    @staticmethod
    def try_add_edge(state: State, grid_position):
        board = state.board
        player = state.turn
        i,  j = grid_position

        if player == PLAYER_1_MARK and state.player_1_walls == 0:
            return False, None
        elif player == PLAYER_2_MARK and state.player_2_walls == 0:
            return False, None

        if not QuoridorBoard.try_edge(state, grid_position):
            return False, None

        logical_position = QuoridorBoard.convert_to_line_position(grid_position)

        if logical_position[0] == 'row':
            for add in range(3):
                board[i][j + add] = WALL
        if logical_position[0] == 'col':
            for add in range(3):
                board[i + add][j] = WALL

        if player == PLAYER_1_MARK:
            state.player_1_walls -= 1
        else:
            state.player_2_walls -= 1

        return True, logical_position

    @staticmethod
    def add_move(state: State, cell):
        """
            Update board based on player turn.
        :param state:
        :param cell: New position
        :return: The old position of the element
        """
        if state.turn == PLAYER_1_MARK:
            player_position = state.player_1_position
            state.player_1_position = cell
        else:
            player_position = state.player_2_position
            state.player_2_position = cell

        return player_position

    def update_player_wall(self, player):
        if player == PLAYER_1_MARK:
            tag = "player_1_wall_count"
            text = f'Player 1 walls: {self.state.player_1_walls}'
            pos = OFFSET + 5 * len(text)
            color = PLAYER_1_COLOR

        else:
            tag = "player_2_wall_count"
            text = f'Player 2 walls: {self.state.player_2_walls}'
            pos = SIZE_OF_BOARD - OFFSET - 5 * len(text)
            color = PLAYER_2_COLOR

        el = self.canvas.find_withtag(tag)
        self.canvas.delete(el)
        self.canvas.create_text(pos,
                                OFFSET / 2,
                                font="cmr 15 bold", text=text, fill=color,
                                tag=tag)

    def update_player_turn(self):
        if self.state.turn == PLAYER_1_MARK:
            text = "Turn: Player 1"
            color = PLAYER_1_COLOR
        else:
            text = "Turn: Player 2"
            color = PLAYER_2_COLOR
        el = self.canvas.find_withtag('player_turn')
        self.canvas.delete(el)
        self.canvas.create_text(OFFSET + 5 * len(text),
                                SIZE_OF_BOARD - OFFSET / 2,
                                font="cmr 15 bold", text=text, fill=color,
                                tag='player_turn')

    # noinspection DuplicatedCode
    def display_edge(self, logical_position, player):
        if logical_position[0] == 'row':
            start_y = (logical_position[1] * DISTANCE_BETWEEN_LINES) + OFFSET
            end_y = start_y
            start_x = (logical_position[2] * DISTANCE_BETWEEN_LINES) + OFFSET
            end_x = start_x + 2 * DISTANCE_BETWEEN_LINES
            start_x -= EDGE_WIDTH / 2
            end_x += EDGE_WIDTH / 2
        elif logical_position[0] == 'col':
            start_x = (logical_position[1] * DISTANCE_BETWEEN_LINES) + OFFSET
            end_x = start_x
            start_y = (logical_position[2] * DISTANCE_BETWEEN_LINES) + OFFSET
            end_y = start_y + 2 * DISTANCE_BETWEEN_LINES
            start_y -= EDGE_WIDTH / 2
            end_y += EDGE_WIDTH / 2
        else:
            return False

        self.canvas.create_line(start_x, start_y,
                                end_x, end_y,
                                fill=WALL_COLOR, width=EDGE_WIDTH, tag='wall')
        self.update_player_wall(player)
        return True

    @staticmethod
    def get_available_moves(state: State):
        if state.turn == PLAYER_1_MARK:
            i, j = state.player_1_position
            player_position = state.player_1_position
            enemy_position = state.player_2_position
        else:
            i, j = state.player_2_position
            player_position = state.player_2_position
            enemy_position = state.player_1_position

        di = [-1, 0, 1, 0]
        dj = [0, 1, 0, -1]
        available_moves = list()
        for k in range(4):
            line_i = i + di[k]
            line_j = j + dj[k]
            if not QuoridorBoard.is_in_board(line_i, line_j):
                continue
            if state.board[line_i][line_j] == LINE:
                cell_i = line_i + di[k]
                cell_j = line_j + dj[k]
                if (cell_i, cell_j) == enemy_position:
                    l_i = cell_i + di[k]
                    l_j = cell_j + dj[k]
                    take_adjacent = False
                    if not QuoridorBoard.is_in_board(l_i, l_j) or state.board[l_i][l_j] == WALL:
                        take_adjacent = True

                    if not take_adjacent:
                        c_i = l_i + di[k]
                        c_j = l_j + dj[k]
                        if state.board[c_i][c_j] == CELL:
                            available_moves.append((c_i, c_j))
                    else:
                        for c in range(4):
                            l_i = cell_i + di[c]
                            l_j = cell_j + dj[c]
                            if not QuoridorBoard.is_in_board(l_i, l_j):
                                continue
                            if state.board[l_i][l_j] == LINE:
                                c_i = l_i + di[c]
                                c_j = l_j + dj[c]
                                if (c_i, c_j) == player_position:
                                    continue
                                available_moves.append((c_i, c_j))
                else:
                    available_moves.append((cell_i, cell_j))

        return available_moves

    @staticmethod
    def get_all_walls(state: State):
        board = state.board
        available_walls = list()
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                if QuoridorBoard.try_edge(state, (i, j)):
                    available_walls.append((i, j))

        return available_walls

    @staticmethod
    def get_all_moves(state: State):
        if state.turn == PLAYER_1_MARK:
            player_walls = state.player_1_walls
        else:
            player_walls = state.player_2_walls

        moves = list()

        player_moves = QuoridorBoard.get_available_moves(state)
        for move in player_moves:
            moves.append(('move', move))
        if player_walls > 0:
            walls = QuoridorBoard.get_all_walls(state)
            for wall in walls:
                moves.append(('wall', wall))

        return moves

    @staticmethod
    def shortest_path(state: State, target_player):
        original_turn = state.turn
        state.turn = target_player
        if target_player == PLAYER_1_MARK:
            player_position = state.player_1_position
            end = 0
        else:
            player_position = state.player_2_position
            end = GRID_SIZE - 1

        expanded_nodes = custom_heap.CustomHeap([(player_position, 0)],
                                                key=lambda x: x[1] + (abs(end - x[0][0])) // 2)
        closed_nodes = list()

        while len(expanded_nodes.data) > 0:
            selected_position = expanded_nodes.pop()

            if selected_position[0][0] == end:
                state.turn = original_turn
                return selected_position[1]

            if target_player == PLAYER_1_MARK:
                original_player_position = state.player_1_position
                state.player_1_position = selected_position[0]
            else:
                original_player_position = state.player_2_position
                state.player_2_position = selected_position[0]

            successors = QuoridorBoard.get_available_moves(state)
            successors_cpy = successors.copy()
            for s in successors_cpy:
                found_in_open = False
                append_to_open = True
                suc = (s, selected_position[1] + 1)
                suc_key = expanded_nodes.gen_key(suc)
                for target in expanded_nodes.data:
                    if suc[0] == target[2][0]:
                        found_in_open = True
                        if suc_key < target[2][1]:
                            expanded_nodes.data.remove(target)
                            expanded_nodes.heapify()
                        else:
                            append_to_open = False
                        break

                if not found_in_open:
                    for close_idx, target in enumerate(closed_nodes):
                        if suc[0] == target[0]:
                            append_to_open = False
                            break
                if append_to_open:
                    expanded_nodes.push(suc)

            closed_nodes.append(selected_position)
            if target_player == PLAYER_1_MARK:
                state.player_1_position = original_player_position
            else:
                state.player_2_position = original_player_position

        state.turn = original_turn
        return -1

    @staticmethod
    def estimate_score_1(state: State):
        final = QuoridorBoard.is_final(state)
        if final:
            if state.turn == PLAYER_1_MARK:
                return - MAX_SCORE - state.depth
            else:
                return MAX_SCORE + state.depth

        ai_score = QuoridorBoard.shortest_path(state, PLAYER_2_MARK)
        enemy_score = QuoridorBoard.shortest_path(state, PLAYER_1_MARK)

        return (enemy_score - ai_score) * 3 + (state.player_1_walls - state.player_2_walls) * 2

    def display_moves(self):
        board = self.state.board
        moves = self.get_available_moves(self.state)
        for cell_i, cell_j in moves:
            logical_position = (cell_i // 2, cell_j // 2)
            start_x = (logical_position[1] * DISTANCE_BETWEEN_LINES) + OFFSET
            end_x = start_x + DISTANCE_BETWEEN_LINES
            start_y = (logical_position[0] * DISTANCE_BETWEEN_LINES) + OFFSET
            end_y = start_y + DISTANCE_BETWEEN_LINES
            self.canvas.create_rectangle(start_x, start_y,
                                         end_x, end_y,
                                         fill=MOVE_COLOR, tag='marked_move')
            board[cell_i][cell_j] = MARKED_MOVE

        self.canvas.tag_raise('wall', 'marked_move')

    def hide_moves(self):
        board = self.state.board
        moves = self.get_available_moves(self.state)
        for cell_i, cell_j in moves:
            board[cell_i][cell_j] = CELL
        self.canvas.delete('marked_move')

    def move_player(self, cell):
        state = self.state
        player_position = self.add_move(self.state, cell)
        if state.turn == PLAYER_1_MARK:
            target = self.player_1
        else:
            target = self.player_2
        x = (cell[1] - player_position[1]) // 2
        y = (cell[0] - player_position[0]) // 2
        self.canvas.move(target, DISTANCE_BETWEEN_LINES * x, DISTANCE_BETWEEN_LINES * y)

    def click(self, event):
        state = self.state
        board = state.board
        abs_position = (event.x, event.y)
        player = PLAYER_1_MARK
        if self.player_move:
            cell = self.convert_to_cell_position(abs_position)
            move = False
            if board[cell[0]][cell[1]] == MARKED_MOVE:
                move = True
            self.hide_moves()
            if move:
                self.move_player(cell)
                self.swap_turn(state)
                self.update_player_turn()
                print(self.shortest_path(state, PLAYER_1_MARK))
            self.player_move = False
        else:
            grid_position = self.convert_to_grid_position(abs_position)
            if grid_position == state.player_1_position:   # clicked on player
                self.display_moves()
                self.player_move = True

            success, logical_position = self.try_add_edge(state, grid_position)

            if not success:
                return

            self.display_edge(logical_position, player)
            self.swap_turn(state)
            self.update_player_turn()

        if state.turn == PLAYER_2_MARK:
            self.canvas.update_idletasks()
            if self.first_move:
                self.move_player((state.player_2_position[0] + 2, state.player_2_position[1]))
                self.first_move = False
            else:
                alpha_beta(-200, 200, state)
                print(state)
                print(type(state.selected_move))
                move, grid_position = state.selected_move
                if move == 'wall':
                    success, logical_position = self.try_add_edge(state, grid_position)
                    self.display_edge(logical_position, PLAYER_2_MARK)
                else:
                    self.move_player(grid_position)
            self.swap_turn(state)
            self.update_player_turn()


@dataclass
class State:
    """
        - Clasa folosita de algoritmii minimax si alpha-beta.
        - O instanta din clasa stare este un nod din arborele minimax.
        - Are ca proprietate tabla de joc.
        - Functioneaza cu conditia ca in cadrul clasei Joc sa fie definiti
        JMIN si JMAX (cei doi jucatori posibili).
        - De asemenea, cere ca in clasa Joc sa fie definita si metoda mutari(),
        care ofera lista cu configuratiile posibile in urma mutarii unui jucator.
    """
    board: Any
    turn: Any
    depth: int
    player_1_position: Any
    player_2_position: Any
    player_1_walls: int
    player_2_walls: int
    parent: Any = None
    score: Any = None
    possible_moves: List[Any] = field(init=False, default_factory=list)
    selected_move: Tuple[Any] = field(init=False, default=None)

    def apply_move(self, move):
        if move[0] == 'wall':
            QuoridorBoard.try_add_edge(self, move[1])
        else:
            QuoridorBoard.add_move(self, move[1])
        QuoridorBoard.swap_turn(self)

    def undo_move(self, move, original_player_position):
        QuoridorBoard.swap_turn(self)
        if move[0] == 'wall':
            QuoridorBoard.undo_edge(self, move[1])
        else:
            QuoridorBoard.add_move(self, original_player_position)


# TODO: add minimax]
def alpha_beta(alpha, beta, state: State):
    if state.depth == 0 or QuoridorBoard.is_final(state):
        state.score = QuoridorBoard.estimate_score_1(state)
        return state

    if alpha > beta:
        return state  # este intr-un interval invalid deci nu o mai procesez

    state.possible_moves = QuoridorBoard.get_all_moves(state)

    if state.turn == PLAYER_2_MARK:
        original_player_position = state.player_2_position
        current_score = float("-inf")

        for move in state.possible_moves:
            # calculeaza scorul
            new_state = copy.copy(state)
            new_state.apply_move(move)
            new_state.depth -= 1
            alpha_beta(alpha, beta, new_state)
            new_state.undo_move(move, original_player_position)
            if current_score < new_state.score:
                state.selected_move = move
                current_score = new_state.score
            if alpha < new_state.score:
                alpha = new_state.score
                if alpha >= beta:
                    break
    else:
        original_player_position = state.player_1_position
        current_score = float("inf")

        for move in state.possible_moves:
            new_state = copy.copy(state)
            new_state.apply_move(move)
            new_state.depth -= 1
            alpha_beta(alpha, beta, new_state)
            new_state.undo_move(move, original_player_position)
            if current_score > new_state.score:
                state.selected_move = move
                current_score = new_state.score

            if beta > new_state.score:
                beta = new_state.score
                if alpha >= beta:
                    break

    state.score = current_score


if __name__ == "__main__":
    game_instance = QuoridorBoard()
    game_instance.mainloop()
