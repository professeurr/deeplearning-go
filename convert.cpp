#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <list>
#include <float.h>
#include <vector>
#include <string>
#include <set>
#include <algorithm>

using namespace std;

#include "Board.h"
#include "Game.h"

int main () {
  loadGames ("games.txt");
  writeGamesData ("games.data");
}
