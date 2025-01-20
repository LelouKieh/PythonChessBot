[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-7f7980b617ed060a017424585567c406b6ee15c891e84e1186181d67ecf80aa0.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=15053699)
# Homework - Adversarial Search ♔♕♗♘♙♖

Topics: Minimax and AlphaBeta

For this assignment you will be making your own chessbot taking advantage of the search techniques discussed in class. You do not need to program the rules of chess in order to complete this assignment.

## Part 0 - Pre-req

There are some libraries and other software that you will need in order to get started.

### Needed Python Packages

* chess - [pypi.org/project/chess/](https://pypi.org/project/chess/) used for modeling boards, identifying legal moves, and faciliating communication. Install with the command `pip install chess`
* pyinstaller - [pyinstaller.org/](https://pyinstaller.org/) for converting your .py files into .exe executables. Install with the command `pip install pyinstaller`
* chester - [pypi.org/project/chester/](https://pypi.org/project/chester/) runs tournaments of chessbots installed with `pip install chester`
* NetworkX - [networkx.org](https://networkx.org/) for graph generation and rendering. Documentation can be found [here](https://networkx.org/documentation/stable/tutorial.html). To install, use the command `pip install networkx`

```bash
pip install chess pyinstaller chester networkx
```

### Visualizing Games

You can use any visualizer you like to play against an engine. The one we'll recommend is Python Easy Chess GUI (see instructions below) which requires some additional setup.

* PySimpleGUI [github.com/PySimpleGUI/PySimpleGUI](https://github.com/PySimpleGUI/PySimpleGUI) creates a generic GUI. Install with `pip install pysimplegui`
* Pyperclip [github.com/asweigart/pyperclip](https://github.com/asweigart/pyperclip) allows for copy/paste functionality with the GUI. Install with `pip install pyperclip`
* Python Easy Chess GUI [https://github.com/fsmosca/Python-Easy-Chess-GUI](https://github.com/fsmosca/Python-Easy-Chess-GUI) clone this repository in a second directory and run the command `python python_easy_chess_gui.py` to run the program.

When setup correctly, it will look like:

![example gui visual](python_easy_chess_gui.png)

```bash
pip install pysimplegui pyperclip
python python_easy_chess_gui.py
```

It is a good idea to turn off Book Moves (Book > Set Book > Uncheck "Use book") and to limit the depth of the chessbots (Engine > Set Depth > 12) so that some bots don't spend all their time thinking. To add a chessbot, go to Engine > Manage > Install > Add > then select your .exe executable. Simply select the opponent by going to Engine > Set Engine Opponent > and select your bot. When ready to play, click Mode > Play. Visit [https://lczero.org/play/quickstart/](https://lczero.org/play/quickstart/) for other visualizers.

### Engines and Tournaments

To create your executable agent use the command `pyinstaller --onefile random_chess_bot.py` except replace with your agent file. This will create an executable, like `random_chess_bot.exe`, inside of a new directory called `dist`. For simplicity, move this file to the directory with the tournament code. **If you are on Mac**, there is another way to make this program executable by using `chmod +x random_chess_bot.py` in the terminal, but pyinstaller should work as well.

In order to test your agent, you'll need to run it against at least one other strong chessbot executable. Good candidates include:

* Stockfish - recommended and probably the strongest open source chessbot [https://stockfishchess.org/](https://stockfishchess.org/) **if you are on mac** you can install using the command `brew install stockfish` and then you should be able to simply run the command `stockfish` to start the bot.
* Goldfish - [https://github.com/bsamseth/Goldfish](https://github.com/bsamseth/Goldfish)
* Leela Chess Zero (Lc0) - [https://lczero.org/](https://lczero.org/)

We recommend downloading the executable to the same directory as the chester tournament code. Edit the [tournament.py](tournament.py) file to add your chessbot as a player. You can then run a tournament with `python tournament.py` and wait for the results.

### How Chess Package Works

If you run the following Python code you'll see the output below it.

```python
import chess

board = chess.Board()
print(board)
```

```text
r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
P P P P P P P P
R N B Q K B N R
```

This is an 8x8 chessboard with the capital letters representing White and lower case for Black. The letter 'P' is for Pawn, 'R' for Rook, 'N' for Knight (not 'K'), 'Q' for Queen, and 'K' for King. The columns are represented with the letters 'a', 'b', 'c', ..., 'h' and rows with the numbers 1 through 8. This means that to give the move Knight on b1 to the spot c3, it is given with the notation Nc3.

The library is able to determine what are the possible valid legal moves allowed by the game with the command `board.legal_moves` which at the start gives:

```text
<LegalMoveGenerator at 0x2283a4b3e80 (Nh3, Nf3, Nc3, Na3, h3, g3, f3, e3, d3, c3, b3, a3, h4, g4, f4, e4, d4, c4, b4, a4)>
```

You can find lots of documentation about all of the functions built into the Python chess library [https://python-chess.readthedocs.io/en/latest/core.html](https://python-chess.readthedocs.io/en/latest/core.html).

If during the development process you wish to visualize the board state like a more traditional image, take a look at `chess.svg` rendering [https://python-chess.readthedocs.io/en/latest/svg.html](https://python-chess.readthedocs.io/en/latest/svg.html). For example, the following code creates the resulting svg graphic.

```python
import chess
import chess.svg

b = chess.Board()
svg = chess.svg.board(b)
f = open("board.svg", "w")
f.write(svg)
f.close()
```

![chess board](board.svg)

You can add images to NetworkX graphs if you like, see [networkx.org/documentation/](https://networkx.org/documentation/stable/auto_examples/drawing/plot_custom_node_icons.html) for more info.

## Part 1 - Instructions

This assignment is meant to ensure that you:

* Understand the concepts of adversarial search
* Can program an agent to traverse a graph along edges
* Experience developing different pruning algorithms
* Apply the basics of Game Theory
* Can argue for chosing one algorithm over another in different contexts

You are tasked with:

0. Copy [random_chess_bot.py](random_chess_bot.py) and update it to develop a new brand new and intelligent chessbot with a unique & non-boring name. ***Do not name it `my_chess_bot`, your name, or something similar.*** If you do, you will ***automatically earn a zero*** for this assignment. Come up with something creative, humourous, witty, adventuous, -- or something will strike fear into the hearts of the other chessbots in this competition.
1. Develop a strong evaluation function for a board state. Take a look at "Programming a Computer for Playing Chess" by Claude Shannon [https://www.computerhistory.org/chess/doc-431614f453dde/](https://www.computerhistory.org/chess/doc-431614f453dde/) published in 1950. You will specifically want to take a look at section 3 in which Shannon describes a straight-forward evaluation function that you can simplify to only evaluate material (pieces) to score a board state.

  * **Note** that your evaluation function will play a crutial role in the strength of your chessbot. It is ok to start with a simple function to get going, but you will need to find ways to improve it because your bot will be competing with the bots from the rest of the class and points are on the line.
  * Talk the teaching team for helpful tips if you are really stuck.
  
2. Implement the Minimax and AlphaBeta pruning algorithms that utilize NetworkX graphs/trees. Visit [algorithm.md](algorithm.md) for more info.
3. Alter your chessbot so that when called with the command line parameter `draw` (such as `python random_chess_bot.py draw`) it creates a Minimax visualization that:

* Starts with the root as the end of a named opening sequence such as the Queen's Gambit Declined 1. d4 d5 2. c4 e6 [https://en.wikipedia.org/wiki/Queen%27s_Gambit_Declined](https://en.wikipedia.org/wiki/Queen%27s_Gambit_Declined). This is because in order for a simple evaluation function to have any chance, there needs to be the potential for pieces to be captured. If you don't like the QGD, we can suggest the:
  * [Ruy Lopez - Morphy Defence](https://en.wikipedia.org/wiki/Ruy_Lopez) 1. e4 e5 2. Nf3 Nc6 3. Bb5 a6
  * [Four Nights Sicilian Defence](https://www.chess.com/openings/Sicilian-Defense-Four-Knights-Variation) 1.e4 c5 2.Nf3 e6 3.d4 cxd4 4.Nxd4 Nf6 5.Nc3 Nc6
  * [Vienna Game Frankenstein–Dracula Variation](https://en.wikipedia.org/wiki/Vienna_Game,_Frankenstein%E2%80%93Dracula_Variation) 1. e4 e5 2. Nc3 Nf6 3. Bc4 Nxe4
  * Any other opening you like that ends with Black making a move so that it is White's turn.
* Have your graph select the top three moves per node and label each edge with the move's notation.
* Limit the depth of the generated tree visuals to four (4) half-moves ahead (W-B-W-B). This is because the visuals will be too difficult to read otherwise.
* Label the leaf nodes with the result of that board state's evaluation
* Perform the Minimax algorithm on the tree, labeling each node backpropogating with the correct minimax value.
* Identify the final value of the game tree and the move that your bot will select in a title or subtitle.
* Perform Alpha-Beta pruning on this game tree to re-color edges and subtrees that have been pruned.
* Finally, draw on the image (use a tablet or print and mark on it) with the results of alpha and beta for each node -- clearly identifying the why & how your graph pruned these edges that it pruned.
* If no branches were pruned, change your opening and/or your evaluation function so that there is some demonstrable pruning.

4. At any given point in a chess game there are roughly 20 possible moves. Your Minimax and Alpha-Beta Pruning algorithms will spend a lot of time on what are clearly poor moves. You are allowed alter these algorithms slightly to not even consider poor quality moves or to only look at the top 7 to 10 moves at a time.
5. When you are done, answer the questions in the reflection and complete the last two sections.

### Example Game Trees

You can use the following code to print the structure of a game tree:

```python
    G = gen.exampleGameTree()
    print(nx.forest_str(G))
    for node in G.nodes():
        print(node,G.nodes[node]["utility"])
```

You may notice the call to `gen.exampleGameTree()`, this creates the tree example given in your textbook. There are other games you can generate using `createRandomGame(rounds, actions, seed=0, isZeroSum=True)` that are completely random i.e. `createRandomGame(3,3,seed=1)` for testing purposes.

Ensure that your chessbot follows normal PyDoc specs for documentation and readability.

## Part 2 - Reflection

Update the README to answer the following questions:

(1) Describe your experiences implementing these algorithms. What things did you learn? What aspects were a challenge? 

I have a really hard time on implementing these algorithms, maybe about 100 to 200 hours were spent in this assignment. I encountered 4 major problems: 

1. Tree generation. At first, I chose to use current fen as the node of the tree, but it does not work because sometimes different moves could lead to the same fen, so some nodes are not generated and we can see some leaf nodes pointed by more than 1 edge. Finally I need to use move sequence as node to solve this problem.
2. Everything messed up and nowhere to debug, I even did not have any ideas about how to debug all these things other than running the tournament praying. Because literally speaking we do not have a tree generation step in the assignment, so after a very simple evaluation function, I skipped to the implementation of these 2 algorithms, and that is a total disaster. Thanks to Prof. Oscar's help I jumped to the tree generation step, but at this time I had spent a lot of time trying to debug all messed up algorithms and trees.
3. How this generated tree interacts with these two algorithms, or how you generated the tree. I was following this procedure: First, generate a tree based on the selected moves. Then, use minimax to update the tree, and here comes the problem: If you use alpha beta pruning with your minimax, then pruning will mess up the tree because when your generated tree does not expect a pruning, it is a full tree with and if you remove some edges and nodes, then problems come to:
4. Visualize pruning. What I do now is using minimax to update the generated tree first, then doing alpha beta pruning with the minimax updated tree. But I believe there are still problems because with debugger and logs I noticed some unusual behaviors: When an edge, which is not an edge not to the leaf is pruned, the tree traversing process will immediately quit after the pruning. I guess that the reason might be still connected with separating minimax and alpha beta pruning, but I do not have time to prove it. Finally I have an idea about this: maybe we can come back to do minimaxing and alpha beta pruning while generating the tree, which is the similar situation that makes me stuck at the first place.
5. Through my hard time, I have learned a lot of things not only about the implementation of algorithms, but also about debugging techinques and how to start with a complicated project in a smart way.

(2) These algorithms assumed that you could reach the leaves of the tree and then reverse your way through it to "solve" the game. In our game (chess) that was not feasible. How effective do you feel that the depth limited search with an evaluation function was in selecting good moves? If you play chess, were you able to beat your bot? If so, why did you beat it? If not, what made the bot so strong - the function or the search?

1. I feel generally speaking in most of the cases, it is relatively effective and could beat random bot easily(around 20 moves or even fewer) in most of the cases. However, when the opponent has only 1 king left, and the king is trying to stay at the corner of the bottom part of its original home, these algorithms are not effective any longer and make some strange move over and over, but ignore those which have a high chance of checkmate. I guess this is because our evaluation function in the given depth gives all the moves very similar evaluation points, for example, all possible moves are the same as 100, so it just acts like(or even worse than because it keeps picking the same piece over and over) selecting random moves.
2. I do not think the depth of search really matters a lot because when I was having a very simple evaluation fucntion and it had some ties with random chess bot, I tried to increase the depth, but it did not actually work and the draw rate did not have an obvious change if we ran 100 or 200 matches. It is the evaluation function that makes the bot really strong, but search depth could also help a little.
3. I do not play chess, I play chess just like the random bot. 

(3) Shannon wrote "... it is possible for the machine to play legal chess, merely making a randomly chosen legal move at each turn to move. The level of play with such a strategy in unbelievably bad. The writer played a few games against this random strategy and was able to checkmate generally in four or five moves (by fool's mate, etc.)" Did you try playing the provided random chessbot and if so, what this your experience? How did your chessbot do against the random bot in your tests?

I do not play chess and I do not play it with random bot. My chess bot can beat the random bot in 5 to 8 moves sometimes. In most cases, it takes my bot around 20 to 30 steps to win. My bot actively eats the pieces from the chess bot: When my bot successfully beats random bot in a few moves, there is a piece entering the opponent's side so it could check the king because random bot knows nothing about defence. If unfortuantely, that one or pieces in the opponent's side are eating by the random moves, it will take longer time. The most tricky situation happens when random bot only has its king left in a corner or their home side, my bot get stuct with one or two pieces and do nothing towards checkmate. 

(4) Explain the what would happen against an opponent who tries to maximize their own utility instead of minimizing yours.

The moves might be more predicable because you know they are always maximizing their utility and less aggressive especially if you consider interruption of your own strategy. Your own strategy is less likely to be distrupted since the opponent only focuses on their maximum outcomes. They might ignore some potential danger in your strategy and expose their king to an open position

(5) What is the "horizon" and how is it used in adversarial tree search?

In adversarial tree search, there is a term called "Horizon Effect", which means you are just "below" the depth to which the tree has been expanded. Horizon refers to the depth of the search tree up to when the game state is evaluated. It sets the limit for the number of different states or possibilities that we could evaluate when giving a problem. This decides how far ahead our AI algorithms could look into when evaluating and choosing the final move. In this assignment, when we implement the minimax algorithms with alpha beta pruning, we examine all the possible moves in the horizon. When we are generating the tree, we are expanding our tree as well as our horizon for our bot.

(6) (Optional) What did you think of this homework? Challenging? Difficult? Fun? Worth-while? Useful? Etc.?
I had such a hard time spending many hours on thinking, debugging and refactoring. It is a mixture of misery, happiness, madness and peace when I finally finshed all parts and sit here to write some optional things. To be honest, my daily routine was completely "destroyed" by the bot for at least a whole week. I am not complaining about it, and this is indeed the most challenging project I have ever implemented as an aligner. This is the first time I have ever been afraid of leaving some parts of my assignment undone (fortunately I made it at the last minute).

As a complete beginner in AI, my suggestions and some thoughts are:
* Reorder the steps of this assignment. If we start with a basic tree generation, it would be easier to understand things and help us to gain confidence with at least some progress.
* More guidance on how to visualize a tree
* I think tree generation is the most important part of the assignment, which need more emphasized and maybe a little bit more detailed explanation. You know your algroithm is running right or wrong mainly through trees. My situation was I implemented something and then tried to run my bot, it lost to the random bot, and I was completely stuck and lost. Evaluation function is not important at the start, we can build a really easy one with only sum for material pieces, it does not guarantee a 100% win but in most cases it works. At first I spent a lot of time doubting whether my evaluation function was too weak or my algorithms were going wrong.
* Actually there are 2 ways as far as I know to generate the trees: Build a tree - Update it with Minimax, and build a tree while implementing minimax and alpha beta pruning. The first one is beginner friendly, but it will have problems when you trying to implement it with alpha beta pruning. Following the previous steps, I was trying to updating the tree updated by minimax with alpha beta pruning, but I don't think it is the right way and I guess we must combine minimax and alpha beta pruning together. There were lots of chaos in this stage for me, and I finally switched back to generating the tree while minimaxing with alpha beta pruning because after dealing with it several days, the complicated logic is not a problem for me, finding one way to prune(remove) the edges and nodes based on the updated minimax tree, then adding them back in visuals seems to be more difficult.
* If more general tips on debugging could be kindly provided along with the instructions of this HW.
---

### Your Images Here

Add the original images that you created and the one that you marked up to clearly demonstrate Alpha-Beta Pruning

Minimax Tree Without Alpha Beta Pruning
![image](https://github.com/CS5150-GameAI-SEA-VAN-Summer24/chess-LelouKieh/assets/18749750/82fae456-cb4f-414f-a2c1-c7f9fb8ca1bb)

Minimax Tree With Alpha Beta Pruning
![image](https://github.com/CS5150-GameAI-SEA-VAN-Summer24/chess-LelouKieh/assets/18749750/f89d7970-8bdf-4c71-adb7-180dffb39eef)

Notes:
* All pruned nodes are shown with an "X", all pruned edges are in red.
* The nodes in green shows that it selects the maximum value from its children.
* The nodes in red shows that it selects the minimum value from its children, or it is a leaf node.
* The blue node is the node for best move.
* The blue line is the selected best move.

How and Why these edges are pruned?
We can refer to the first picture, Minimax Tree Without Alpha Beta Pruning. Firstly, we traverse the tree from the blue node(root), then we go to c4d5(20, step 1), then we further visit 20(e6d5, step 2), and finally reached the 3 leaves of 20. Then we go back to 20(c4d5), and visit 30(step 4). Then we visit the leaves of 30, we get a 30 through one leaf(step 5), and the other 2 are less than 30, we are doing maximizing in this level so we could prune them(step 6). We do this repeatedly until all leaves are visited or pruned. You can see the details step in the picture below:

<img width="1435" alt="AB pruning with steps" src="https://github.com/CS5150-GameAI-SEA-VAN-Summer24/chess-LelouKieh/assets/18749750/ef184aa5-8ca1-4238-ac15-9506f7d986a4">


### Your Evaluation Function Here

Conciesly and effictively describe the evaluation function that you used for your chessbot. You can also use Latex as long as you explain the symbols and justify why you created your function in the manner with which you did.

My evaluation function can be broken into 3 parts: Material Value Evaluation, Positional Value Evaluation and Special Phases. I refer to other's evaluation functions online to complete the whole logic because I am a completely chess beginner and know very little about chess startgies, I will list them as references by the end.

#### Material Evaluation
Each type of pieces has its own value. We will give a higher value for more important pieces like the Queen. We will give a really large value for the King, and a small value for normal pieces like pawns. When we examine the chess board, we will firstly calculate each side's remaining pieces with their sum of material values based on their types.

#### Positional Value
This is a very stratgetic part. When a piece is at the different positions of the board, it has different potentials. For example, if a knight is put in the middle part of the chess board, it will have a higher chance to eat opponent's pieces based on its moving pattern. If it is placed at the edge of board or just its home, it can hardly do any checks. Therefore, for each type of pieces, we need to assign them a positional value based on their positions on the board. If we want to encourage knights to enter the middle part, these squares of board should be assigned with higher scores, and the edge squares should be assigned with lower scores. We calculate the sum of positional values for all pieces on the board from each side, add the sums of both material values and positional values together for both sides.

#### Special Phases
We have some special phases like checkmate and draws, we need to assign them with a super high value or a 0 value because we want checkmate and do not want draws. And I also add an end phase check for kings: when the game just starts, we do not want our kings to move too far because this makes the King in danger and easily checked by so many pieces. However, when there are very few pieces on the board with few obstacles to prevent these pieces with long traveling distances like rook, so the king could move around a little bit and stay at home is no longer the best strategy. 

So there is an ending phase calculator for the kings. We have a initial phase score. When a piece is eaten, based on what type of the piece it is, we will subtract the phase score with the piece's phase constrant. When the phase score drop below half of its original value, we enter the end game phase. Notice that pawns are not included because it is not considered as powerful and it travels only a short distance.

When we enter the end game phase, our king is given a different set of positional values, which encourages it move around and not just staying at home. When calculating positional value, we need to use the end game set for kings.

Generally speaking, my evaluation function can be concluded as:

$$\text{Evaluation points for One Side} = \sum \text{Material values} + \sum \text{Positional Values}$$

We also need to use our side of this sum to subtract the opponent's sum to get the final evaluation value:

$$\text{Evaluation Value} = \sum \text{My Bot's Points} - \sum \text{Opponent's Points}$$

#### References:
https://github.com/gautambajaj/Chess-AI/blob/master/js/chessAI.js

https://github.com/lamesjim/Chess-AI

https://github.com/emdio/secondchess/blob/master/secondchess.c
