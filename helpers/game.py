import os
import re

import cv2
import pygame

from .move_detection import RockPaperScissorsPredictor, MovesEnum
from .next_move_prediction import NextMovePredictor
from .webcam import opencv_video_capture, opencv_to_pygame_image

base_path = os.getcwd()


class Game:
    """
    This class contains all the game logic
    """

    # Dict that contains the images of the moves
    BOT_MOVE_IMAGES = {
        MovesEnum.ROCK: pygame.image.load(
            os.path.join(base_path, "assets", "img", "rock.png")
        ),
        MovesEnum.PAPER: pygame.image.load(
            os.path.join(base_path, "assets", "img", "paper.png")
        ),
        MovesEnum.SCISSORS: pygame.image.load(
            os.path.join(base_path, "assets", "img", "scissors.png")
        ),
    }

    # Icon image of the window
    ICON_IMAGE = pygame.image.load(os.path.join(base_path, "assets", "img", "icon.png"))
    # playingwith.ai logo image
    LOGO_IMAGE = pygame.image.load(os.path.join(base_path, "assets", "img", "logo.png"))
    # "VS" image
    VS_IMAGE = pygame.image.load(os.path.join(base_path, "assets", "img", "vs.png"))
    # Robot image that will be displayed before a bot move
    ROBOT_IMAGE = pygame.image.load(
        os.path.join(base_path, "assets", "img", "question.png")
    )

    # Colors declaration
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    PLAYINGWITHAI_COLOR = (66, 153, 225)
    PLAYINGWITHAI_DARK_COLOR = (56, 143, 215)

    # Table of the points
    point_table = [
        [0, -1, 1],
        [1, 0, -1],
        [-1, 1, 0],
    ]

    def __init__(
        self,
        screen_width=800,
        screen_height=600,
        webcam_index=0,
        min_repeated_move_detection=30,
        no_detection_period=5,
    ):
        # create dir that contains all the data of this project
        self._create_data_dir()
        # Game window resolution
        self.screen_width = screen_width
        self.screen_height = screen_height
        # Index of the webcam to use (0 is the default one)
        self.webcam_index = webcam_index
        # In order to prevent wrong detections,
        # the same move must be detected "min_repeated_move_detection" times
        self.min_repeated_move_detection = min_repeated_move_detection
        # A period during which there's no move detection
        self.no_detection_period = no_detection_period
        # Font for title
        self.font_title = self._init_font(36)
        # Font for small text
        self.font_small = self._init_font(16)
        # Reset current score
        self.current_score = 0
        # Reset open cv camera acquisition
        self.camera = None
        # Reset the last move of the bot
        self.last_bot_move = None
        # Reset the last move of the user
        self.last_user_move = None
        # How many times a move is repeatedly detected
        self.repeated_move_detection_counter = 0
        # Seconds with no move detection between rounds
        self.no_detection_rounds = self.no_detection_period
        # Init of the user next move predictor built in the second part of this tutorial
        # https://playingwith.ai/blog/morra-cinese-contro-ia-parte2.html
        self.user_next_move_predictor = NextMovePredictor()
        # Init of the move detector built in the first part of this tutorial:
        # https://playingwith.ai/blog/morra-cinese-contro-ia-parte1.html
        self.move_detector = RockPaperScissorsPredictor()
        # Prevent tensorflow to load during the first detection
        self.move_detector_load_needed = True
        # Main cycle variable, if False the game will quit
        self.running = True
        # True if a user is playing
        self.playing = False
        # True when user lost the game
        self.lost = True
        # If true hand detection must be stopped
        self.stop_detection = False
        # The last round user point
        self.last_user_point = None
        # Path for the file that contains the high score
        self.score_dir_path = os.path.join(base_path, "data", "score")
        # Get high score from stored file
        self.high_score = self._get_high_score()
        # init pygame
        self._init_pygame()

        # Sound constant must be declared after pygame init or an exception will be
        # raised

        # Sound for round win
        self.SOUND_WIN = pygame.mixer.Sound(
            os.path.join(base_path, "assets", "audio", "you_win.wav")
        )
        # Sound for round draw
        self.SOUND_DRAW = pygame.mixer.Sound(
            os.path.join(base_path, "assets", "audio", "draw.wav")
        )
        # Sound for round lost
        self.SOUND_LOST = pygame.mixer.Sound(
            os.path.join(base_path, "assets", "audio", "you_lose.wav")
        )
        # Sound for round start
        self.SOUND_FIGHT = pygame.mixer.Sound(
            os.path.join(base_path, "assets", "audio", "fight.wav")
        )
        # This variable prevent the round start sound to be played more than once per
        # round
        self.PLAY_FIGHT = True
        # Sound 3 second countdown
        self.SOUND_3 = pygame.mixer.Sound(
            os.path.join(base_path, "assets", "audio", "3.wav")
        )
        # This variable prevent the 3 second countdown sound to be played more than
        # once per round
        self.play_3 = True
        # Sound 2 second countdown
        self.SOUND_2 = pygame.mixer.Sound(
            os.path.join(base_path, "assets", "audio", "2.wav")
        )
        # This variable prevent the 2 second countdown sound to be played more than
        # once per round
        self.play_2 = True
        # Sound 1 second countdown
        self.SOUND_1 = pygame.mixer.Sound(
            os.path.join(base_path, "assets", "audio", "1.wav")
        )
        # This variable prevent the 1 second countdown sound to be played more than
        # once per round
        self.play_1 = True

    def _get_high_score(self):
        # Get high score from score.txt file if exists
        score_file_path = os.path.join(self.score_dir_path, "score.txt")
        if not os.path.exists(score_file_path):
            return 0
        with open(score_file_path, "r") as f:
            score = re.findall(r"\d+", f.read())
            return score and int(score[0]) or 0

    def _set_high_score(self):
        # Save high score into score.txt file if it's higher
        if self.high_score < self._get_high_score():
            return

        if not os.path.exists(self.score_dir_path):
            return os.mkdir(self.score_dir_path)
        with open(os.path.join(self.score_dir_path, "score.txt"), "w") as f:
            f.write(str(self.high_score))

    @staticmethod
    def _init_font(size=24):
        # pygame font init
        pygame.font.init()
        return pygame.font.Font(
            os.path.join(base_path, "assets", "fonts", "font.ttf"), size
        )

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
        move_predictor_path = os.path.join(data_path, "move_detector")
        if not os.path.exists(move_predictor_path):
            os.mkdir(move_predictor_path)

    def _set_screen(self):
        # Set pygame window resolution
        return pygame.display.set_mode((self.screen_width, self.screen_height))

    def _set_window_icon_and_title(self):
        # Set pygame windows icon and title
        pygame.display.set_icon(self.ICON_IMAGE)
        pygame.display.set_caption("Rock Paper Scissors against an AI")

    @staticmethod
    def _set_one_second_timer():
        # pygame 1s timer
        pygame.time.set_timer(pygame.USEREVENT, 1000)

    def _init_pygame(self):
        # required by pygame
        pygame.init()
        # set the logo and title
        self._set_window_icon_and_title()
        # set the pygame screen resolution
        self.screen = self._set_screen()
        # set a custom 1s timer
        self._set_one_second_timer()

    def _show_centered_text(
        self, caption, font, frame_width, height, color, width_span=0.0
    ):
        # An helper method that displays a text centered based on variable frame_width
        text = font.render(caption, True, color)
        text_rect = text.get_rect()
        text_rect.center = ((frame_width / 2) + width_span, height)
        self.screen.blit(text, text_rect)

    def _show_logo(self, position):
        # Show logo in pygame window
        self.screen.blit(self.LOGO_IMAGE, position)

    def _new_game(self):
        # Restart a game resetting variables
        self.playing = True
        self.lost = False
        self.last_user_move = None
        self.last_bot_move = None
        self.no_detection_rounds = self.no_detection_period

    def _show_start_game_button(self, x=355, y=500, width=90, height=50):
        # Start game button is visible only if we're playing the first game or the
        # game is lost
        if self.lost:
            # get the mouse position
            mouse = pygame.mouse.get_pos()
            # get if mouse is pressed
            click = pygame.mouse.get_pressed()
            # if mouse is over the start button
            if x + width > mouse[0] > x and y + height > mouse[1] > y:
                # Start button will be rendered with a darken color
                pygame.draw.rect(
                    self.screen, self.PLAYINGWITHAI_DARK_COLOR, (x, y, width, height)
                )
                # If start button is clicked, start a new game
                if click[0] == 1:
                    self._new_game()
            else:
                # No mouse over the start button, default rendering
                pygame.draw.rect(
                    self.screen, self.PLAYINGWITHAI_COLOR, (x, y, width, height)
                )
            # Display the "Play" text inside the start button
            self._show_centered_text(
                "Play", self.font_title, self.screen_width, 525, self.WHITE
            )

    def _show_bot_move_element(self):
        # Display "Computer" text over the bot image
        self._show_centered_text(
            "Computer", self.font_title, self.screen_width / 2, 120, self.RED
        )
        # Load the bot move if is set, default robot image if not
        image = (
            self.ROBOT_IMAGE
            if self.last_bot_move is None
            else self.BOT_MOVE_IMAGES[self.last_bot_move]
        )
        self.screen.blit(image, (50, 140))

    def _handle_user_move_detection(self, user_webcam_image):
        # Detect the user move from the picture
        move_detected = self.move_detector.detect_move_from_picture(user_webcam_image)
        if move_detected is None:
            # No move is detected -> reset repeated_move_detection_counter
            self.repeated_move_detection_counter = 0
        elif (
            self.last_user_move is not None
            and self.last_user_move == move_detected.value
        ):
            # Last user move is equal to the current move detected -> increase
            # repeated_move_detection_counter
            self.repeated_move_detection_counter += 1
        else:
            # Move detected is not the same of the user last move -> reset
            # repeated_move_detection_counter and store the new user move
            self.repeated_move_detection_counter = 0
            self.last_user_move = move_detected

        # If a move is detected repeatedly for more than
        # min_repeated_move_detection -> stop the move detection and play a round
        if self.repeated_move_detection_counter > self.min_repeated_move_detection:
            self.stop_detection = True
            self._play_round()

    def _handle_user_image_acquisition_and_detection(self):
        # Get webcam frame
        _, user_webcam_image = self.camera.read()
        if self.move_detector_load_needed:
            # Tensorflow needs a lot of time for the init, so we do a false detection to
            # load it in the first cycle
            self.move_detector.detect_move_from_picture(user_webcam_image)
            self.move_detector_load_needed = False
        # Move detection must be done only if user is playing, if detection is
        # allowed and if no_detection_rounds are less or equal 0
        if self.playing and not self.stop_detection and self.no_detection_rounds <= 0:
            self._handle_user_move_detection(user_webcam_image)

        # pygame needs some image conversion to properly display the frame acquired
        # with opencv
        return opencv_to_pygame_image(user_webcam_image)

    def _show_user_move_element(self):
        # Display "You" text over the user image
        self._show_centered_text(
            "You",
            self.font_title,
            self.screen_width / 2,
            120,
            self.PLAYINGWITHAI_COLOR,
            width_span=self.screen_width / 2,
        )

        # Acquire webcam image and do move detection
        user_webcam_image = self._handle_user_image_acquisition_and_detection()
        # Display user webcam image in pygame window
        self.screen.blit(user_webcam_image, (450, 140))

    def _show_high_score(self):
        # Display high score in pygame window
        text = self.font_title.render("High score:", True, self.GREEN)
        self.screen.blit(text, (50, 450))
        text = self.font_small.render(f"{self.high_score}", True, self.BLACK)
        self.screen.blit(text, (230, 465))

    def _show_current_score(self):
        # Display current score in pygame window
        text = self.font_title.render("Current score:", True, self.PLAYINGWITHAI_COLOR)
        self.screen.blit(text, (450, 450))
        text = self.font_small.render(f"{self.current_score}", True, self.BLACK)
        self.screen.blit(text, (670, 465))

    def _show_vs_image(self):
        # display vs image in pygame window
        self.screen.blit(self.VS_IMAGE, (380, 280))

    def _show_round_countdown(self):
        # Display the round countdown in the user webcam frame
        if self.playing and self.no_detection_rounds > 0:
            self._show_centered_text(
                str(self.no_detection_rounds),
                self.font_title,
                self.screen_width / 2,
                300,
                self.WHITE,
                width_span=self.screen_width / 2,
            )

    def _show_gui_elements(self):
        # calls all methods related to the gui
        self._show_logo((10, 10))
        self._show_centered_text(
            "Rock Paper Scissors", self.font_title, self.screen_width, 60, self.BLACK
        )
        self._show_bot_move_element()
        self._show_user_move_element()
        self._show_round_countdown()
        self._show_start_game_button()
        self._show_high_score()
        self._show_current_score()
        self._show_vs_image()

    def _quit_game(self):
        # Quit the game
        self.running = False

    def _sounds(self):
        # It handles the game sounds
        if self.playing:
            if self.PLAY_FIGHT and self.no_detection_rounds == 0:
                self.SOUND_FIGHT.play()
                self.PLAY_FIGHT = False
            elif self.play_3 and self.no_detection_rounds == 3:
                self.SOUND_3.play()
                self.play_3 = False
            elif self.play_2 and self.no_detection_rounds == 2:
                self.SOUND_2.play()
                self.play_2 = False
            elif self.play_1 and self.no_detection_rounds == 1:
                self.SOUND_1.play()
                self.play_1 = False

    def _reset_bot_move(self):
        # Every round the last bot move is hidden before the new one is played
        if self.no_detection_rounds == 1:
            self.last_bot_move = None

    def _reset_sounds(self):
        # Reset all sounds at the end of every round
        self.PLAY_FIGHT = True
        self.play_3 = True
        self.play_2 = True
        self.play_1 = True

    def _check_events(self):
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self._quit_game()
            if event.type == pygame.USEREVENT:
                # this event is raised every 1s
                if self.playing and not self.stop_detection:
                    self.no_detection_rounds -= 1

    def _set_background_color(self):
        self.screen.fill(self.WHITE)

    def _get_bot_move(self):
        # Get the move that defeat the predicted user move
        return self.point_table[
            self.user_next_move_predictor.predict_next_move()
        ].index(-1)

    def _update_bot(self, user_move):
        # Train bot with the new user move
        self.user_next_move_predictor.train(user_move)

    def _update_score(self, user_point):
        self.current_score += max(0, user_point)

    def _get_user_round_point(self, user_move, bot_move):
        # return the point that user get in current round
        return self.point_table[user_move][bot_move]

    def _end_game(self):
        self.SOUND_LOST.play()
        self.lost = True
        self.high_score = max(self.high_score, self.current_score)
        self.current_score = 0
        # Save high score if is higher of the old one
        self._set_high_score()
        # Reset the played move for user move predictions
        self.user_next_move_predictor.reset_played_moves()

    def _show_result(self):
        if self.last_user_point is None:
            return

        # Update the user score
        self._update_score(self.last_user_point)
        # if user loses stop the game
        if self.last_user_point == -1:
            self._end_game()
        else:
            # play proper sound
            if self.last_user_point == 1:
                self.SOUND_WIN.play()
            else:
                self.SOUND_DRAW.play()
            # Reset no detection rounds
            self.no_detection_rounds = self.no_detection_period
            # reset repeated_move_detection_counter
            self.repeated_move_detection_counter = 0
            # Game can continue
            self.playing = True
        # Reset last_user_point
        self.last_user_point = None
        self._reset_sounds()
        # detection allowed
        self.stop_detection = False

    def _play_round(self):
        if not self.playing:
            return
        # pause playing
        self.playing = False
        # Get the move that bot wants to play
        self.last_bot_move = self._get_bot_move()
        # Calculate the user point
        self.last_user_point = self._get_user_round_point(
            self.last_user_move, self.last_bot_move
        )
        # Train the bot with the last round
        self._update_bot(self.last_user_move)

    # Main game cycle
    def run(self):
        try:
            with opencv_video_capture(self.webcam_index) as camera:
                self.camera = camera
                while self.running:
                    self._reset_bot_move()
                    self._set_background_color()
                    self._check_events()
                    self._show_gui_elements()
                    self._show_result()
                    self._sounds()
                    pygame.display.update()
        finally:
            self.user_next_move_predictor.save_model()
