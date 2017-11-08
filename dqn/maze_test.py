from maze_env import Maze
import random

myMaze= Maze()

observation = myMaze.reset()
s,r,done = myMaze.step(1)

print s
print r
print done

 
s,r,done = myMaze.step(1)
print s
print r
print done

s,r,done = myMaze.step(1)
print s
print r
print done

myMaze.render()
    
