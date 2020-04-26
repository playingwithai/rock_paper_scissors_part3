import os
import re

import cv2
import pygame

from .move_detection import RockPaperScissorsPredictor, MovesEnum
from .next_move_prediction import NextMovePredictor
from .webcam import opencv_video_capture, get_webcam_frame

base_path = os.getcwd()

BOT_MOVE_IMAGES = {
    MovesEnum.ROCK: os.path.join(base_path, "assets", "img", "rock.png"),
    MovesEnum.PAPER: os.path.join(base_path, "assets", "img", "paper.png"),
    MovesEnum.SCISSORS: os.path.join(base_path, "assets", "img", "scissors.png"),
}

MOVE_LABEL = {
    MovesEnum.ROCK: os.path.join(base_path, "assets", "img", "rock.png"),
    MovesEnum.PAPER: os.path.join(base_path, "assets", "img", "paper.png"),
    MovesEnum.SCISSORS: os.path.join(base_path, "assets", "img", "scissors.png"),
}


class Game:
    """
    This class contains all the game logic
    """

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    PLAYINGWITHAI_COLOR = (66, 153, 225)
    PLAYINGWITHAI_DARK_COLOR = (56, 143, 215)

    point_table = [
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0],
    ]

    def __init__(self, screen_width=800, screen_height=600, webcam_index=0):
        self.running = True
        self.webcam_index = webcam_index
        self.screen_width = screen_width
        self.screen_height = screen_height
        self._create_data_dir()
        self._init_pygame()
        self._set_one_second_timer()
        self.font_title = self._init_font(36)
        self.font_small = self._init_font(16)
        # set the logo and title
        self._set_window_icon_and_title()
        self.screen = self._set_screen()
        self.camera = None
        self.score_dir_path = os.path.join(base_path, "data", "score")
        self.high_score = self._get_high_score()
        self.current_score = 0
        self.last_bot_move = None
        self.last_user_move = None
        self.prevent_wrong_move_detection_counter = 0
        self.min_move_repeated_detection = 10
        self.playing = False
        self.no_detection_rounds = 3
        self.move_predictor = RockPaperScissorsPredictor()
        self.user_next_move_predictor = NextMovePredictor()
        self.user_next_move_predictor.load_model()
        self.lost = True

    @staticmethod
    def _create_data_dir():
        data_path = os.path.join(base_path, "data")
        if not os.path.exists(data_path):
            os.mkdir(data_path)
        score_path = os.path.join(data_path, "score")
        if not os.path.exists(score_path):
            os.mkdir(score_path)
        move_detector_path = os.path.join(data_path, "move_detector")
        if not os.path.exists(move_detector_path):
            os.mkdir(move_detector_path)
        move_predictor_path = os.path.join(data_path, "move_predictor")
        if not os.path.exists(move_predictor_path):
            os.mkdir(move_predictor_path)

    @staticmethod
    def _init_pygame():
        pygame.init()


    @staticmethod
    def _set_one_second_timer():
        pygame.time.set_timer(pygame.USEREVENT, 1000)

    def _set_screen(self):
        return pygame.display.set_mode((self.screen_width, self.screen_height))

    def _get_high_score(self):
        score_file_path = os.path.join(self.score_dir_path, "score.txt")
        if not os.path.exists(score_file_path):
            return 0
        with open(score_file_path, "r") as f:
            score = re.findall(r"\d+", f.read())
            return score and int(score[0]) or 0

    def _set_high_score(self):
        if self.high_score < self._get_high_score():
            return

        if not os.path.exists(self.score_dir_path):
            return os.mkdir(self.score_dir_path)
        with open(os.path.join(self.score_dir_path, "score.txt"), "w") as f:
            f.write(str(self.high_score))

    @staticmethod
    def _init_font(size=24):
        pygame.font.init()
        return pygame.font.Font(os.path.join(base_path, "assets", "fonts", "font.ttf"), size)

    @staticmethod
    def _set_window_icon_and_title():
        logo = pygame.image.load(os.path.join(base_path, "assets", "img", "icon.png"))
        pygame.display.set_icon(logo)
        pygame.display.set_caption("Rock Paper Scissors against an AI")

    def _show_centered_text(self, caption, font, frame_width, height, color, width_span=0.0):
        text = font.render(caption, True, color)
        text_rect = text.get_rect()
        text_rect.center = ((frame_width / 2) + width_span, height)
        self.screen.blit(text, text_rect)

    def _show_logo(self, position):
        logo_image = pygame.image.load(os.path.join(base_path, "assets", "img", "logo.png"))
        self.screen.blit(logo_image, position)

    def _show_start_game_button(self, x=355, y=500, width=90, height=50):
        if self.lost:
            mouse = pygame.mouse.get_pos()
            click = pygame.mouse.get_pressed()
            if x + width > mouse[0] > x and y + height > mouse[1] > y:
                pygame.draw.rect(self.screen, self.PLAYINGWITHAI_DARK_COLOR, (x, y, width, height))

                if click[0] == 1:
                    self.playing = True
                    self.lost = False
                    self.last_user_move = None
                    self.last_bot_move = None

            else:
                pygame.draw.rect(self.screen, self.PLAYINGWITHAI_COLOR, (x, y, width, height))
            self._show_centered_text("Play", self.font_title, self.screen_width, 525, self.WHITE)

    def _show_bot_move_element(self):
        if self.last_bot_move:
            bot_move_image = pygame.image.load(BOT_MOVE_IMAGES[self.last_bot_move])
            self.screen.blit(bot_move_image, (50, 140))
        else:
            self._show_centered_text("Computer", self.font_title, self.screen_width / 2, 120, self.RED)
            pygame.draw.rect(self.screen, self.BLACK, (50, 140, 300, 300))

    def _show_user_move_element(self):
        user_webcam_image = get_webcam_frame(self.camera)
        if self.playing and self.no_detection_rounds <= 0:
            move_detected = self.move_predictor.detect_move_from_picture(user_webcam_image)
            if move_detected is None:
                self.prevent_wrong_move_detection_counter = 0
            elif self.last_user_move is not None and self.last_user_move == move_detected.value:
                self.prevent_wrong_move_detection_counter += 1
            else:
                self.prevent_wrong_move_detection_counter = 0
                self.last_user_move = move_detected

            if self.prevent_wrong_move_detection_counter > self.min_move_repeated_detection:
                self._play_round()

        user_webcam_image = cv2.cvtColor(user_webcam_image, cv2.COLOR_BGR2RGB)
        user_webcam_image = pygame.image.frombuffer(user_webcam_image.tostring(), user_webcam_image.shape[1::-1], "RGB")
        self._show_centered_text(
            "You",
            self.font_title,
            self.screen_width / 2,
            120,
            self.PLAYINGWITHAI_COLOR,
            width_span=self.screen_width / 2
        )
        self.screen.blit(user_webcam_image, (450, 140))
        if self.playing and self.no_detection_rounds > 0:
            self._show_centered_text(
                str(self.no_detection_rounds),
                self.font_title,
                self.screen_width / 2,
                300,
                self.WHITE,
                width_span=self.screen_width / 2
            )

    def _show_high_score(self):
        text = self.font_title.render("High score:", True, self.GREEN)
        self.screen.blit(text, (50, 450))
        text = self.font_small.render(f"{self.high_score}", True, self.BLACK)
        self.screen.blit(text, (230, 465))

    def _show_current_score(self):
        text = self.font_title.render("Current score:", True, self.PLAYINGWITHAI_COLOR)
        self.screen.blit(text, (450, 450))
        text = self.font_small.render(f"{self.current_score}", True, self.BLACK)
        self.screen.blit(text, (670, 465))

    def _show_vs_image(self):
        logo_image = pygame.image.load(os.path.join(base_path, "assets", "img", "vs.png"))
        self.screen.blit(logo_image, (380, 280))

    def _show_gui_elements(self):
        self._show_logo((10, 10))
        self._show_centered_text("Rock Paper Scissors", self.font_title, self.screen_width, 60, self.BLACK)
        self._show_bot_move_element()
        self._show_user_move_element()
        self._show_start_game_button()
        self._show_high_score()
        self._show_current_score()
        self._show_vs_image()

    def _quit_game(self):
        self.running = False

    def _check_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit_game()
            if event.type == pygame.USEREVENT:
                if self.playing:
                    self.no_detection_rounds = max(0, self.no_detection_rounds - 1)

    def _set_background_color(self):
        self.screen.fill(self.WHITE)

    def _get_bot_move(self):
        # Get the move that defeat the predicted user move
        return self.point_table[self.user_next_move_predictor.predict_next_move()].index(-1)

    def _update_bot(self, user_move):
        # Train bot with the new user move
        self.user_next_move_predictor.train(user_move)

    def _update_score(self, user_point):
        self.current_score += max(0, user_point)

    def _get_user_round_point(self, user_move, bot_move):
        # return the point that user get in current round
        return self.point_table[user_move][bot_move]

    def _play_round(self):
        if not self.playing:
            return

        self.playing = False
        self.last_bot_move = self._get_bot_move()
        user_point = self._get_user_round_point(self.last_user_move, self.last_bot_move)
        self._update_score(user_point)
        self._update_bot(self.last_user_move)
        if user_point == -1:
            self.lost = True
            self.high_score = max(self.high_score, self.current_score)
            self.current_score = 0
            self._set_high_score()
            self.user_next_move_predictor.save_model()
        else:
            self.playing = True
            self.no_detection_rounds = 3
            self.prevent_wrong_move_detection_counter = 0

    def run(self):
        with opencv_video_capture(self.webcam_index) as camera:
            self.camera = camera
            while self.running:
                self._set_background_color()
                self._check_events()
                self._show_gui_elements()
                pygame.display.update()
