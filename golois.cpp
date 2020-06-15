#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <list>
#include <float.h>
#include <vector>
#include <set>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

using namespace std;

#include "Board.h"
#include "Game.h"

bool loaded = false;

int nbExamples = 128;

PYBIND11_MODULE(golois, m) {
  m.def("getBatch", [](py::array_t<float> x, py::array_t<float> policy,
		       py::array_t<float> value, py::array_t<float> end) {
	if (!loaded) {
	  memcpy (historyBoard [0], board.board, MaxSize);
	  fprintf (stderr, "load games.data\n");
	  loadGamesData ("games.data");
	  loaded = true;
	}
	auto r = x.mutable_unchecked<4>();
	auto pi = policy.mutable_unchecked<2>();
	auto v = value.mutable_unchecked<1>();
 	auto e = end.mutable_unchecked<4>();
	nbExamples = r.shape (0);
	fprintf (stderr, "r.shape = (%d, %d, %d, %d)\n", r.shape (0), r.shape (1), r.shape (2), r.shape (3));
	fprintf (stderr, "nbExamples = %d\n", nbExamples);
	for (ssize_t i = 0; i < nbExamples; i++) {
	  //fprintf (stderr, "i = %d, ", i);
	  // choose a random state
	  int pos = rand () % nbPositionsSGF;
	  //fprintf (stderr, "pos = %d, positionSGF [pos].game = %d, positionSGF [pos].move = %d\n",
	  //	   pos, positionSGF [pos].game, positionSGF [pos].move);
	  Board b = board;
	  for (int j = 0; j < positionSGF [pos].move; j++)
	    play (&b, proGame [positionSGF [pos].game] [j].inter);
	  //fprintf (stderr, "fill the input\n");
	  // fill the input
	  int turn = b.turn;
	  int other = Black;
	  if (b.turn == Black)
	    other = White;
  	  encode (&b);
	  for (ssize_t j = 0; j < r.shape(1); j++)
	    for (ssize_t k = 0; k < r.shape(2); k++)
	      for (ssize_t l = 0; l < r.shape(3); l++) {
		r (i, j, k, l) = input [j] [k] [l];
	      }
	  // fill the policy
 	  for (ssize_t j = 0; j < pi.shape(1); j++)
	    pi (i, j) = 0.0;
	  int move = proGame [positionSGF [pos].game] [positionSGF [pos].move].inter;
	  pi (i, move) = 1.0;
	  // fill the value
	  if (winner [positionSGF [pos].game] == 'W')
	    v (i) = 1.0;
	  else
	    v (i) = 0.0;
	  //fprintf (stderr, "fill the endgame\n");
	  // fill the endgame
	  for (int j = positionSGF [pos].move; j < nbMovesSGFGame [positionSGF [pos].game]; j++) {
	    //fprintf (stderr, "positionSGF [pos].game = %d\n", positionSGF [pos].game);
	    //fprintf (stderr, "nbMovesSGFGame [positionSGF [pos].game] = %d\n", nbMovesSGFGame [positionSGF [pos].game]);
	    //fprintf (stderr, "j = %d\n", j);
	    play (&b, proGame [positionSGF [pos].game] [j].inter);
	  }
	  for (int j = 0; j < 19; j++)
	    for (int k = 0; k < 19; k++) {
	      if (b.board [interMove [19 * j + k]] == turn) {
		e (i, j, k, 0) = 1.0;
		e (i, j, k, 1) = 0.0;
	      }
	      else if (b.board [interMove [19 * j + k]] == other) {
		e (i, j, k, 0) = 0.0;
		e (i, j, k, 1) = 1.0;
	      }
	      else {
		e (i, j, k, 0) = 0.0;
		e (i, j, k, 1) = 0.0;
	      }
	    }
	}
    });
}
