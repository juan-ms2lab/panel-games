# panel-games

Multiplayer LED games where phones are the controllers. Games render into a numpy RGB canvas of their own
size, and the server pushes each frame to one or more outputs:

- **DDP** (Distributed Display Protocol over UDP) to an LED driver such as
  colorlight-driver (the fork of Austin's driver) driving Colorlight 5A-75B panels;
- a **screen** page (`/screen`) for any TV, monitor or laptop browser.

The games (everything except Panel Flap Resurrection) come from Austin's
[Colorlight-LED-Panel-Drivers](https://github.com/Austinrocket/Colorlight-LED-Panel-Drivers) (`ledpanels`),
extracted here with their controller page, player slots, rate limits and tests. Hence the GPLv3 license.

## Quick start

```
uv run panel-games serve
```

Open `http://localhost:8090/screen` on a TV or monitor (full screen it), and on each phone open
`http://<this-computer's-ip>:8090` (the address is printed at start and shown on the idle screen). Pick a game,
press Play, and the phone becomes a controller; more phones press "Join this game".

Options (all optional):

| flag | default | |
|---|---|---|
| `--ddp HOST[:PORT]` | none | also send frames by DDP (port 4048 if omitted) |
| `--target WxH` | `768x128` | output canvas; games that do not fit are listed as unavailable |
| `--port` | `8090` | web port |
| `--bind` | `0.0.0.0` | web address (all interfaces) |
| `--fps` | `60` | frames sent per second (game logic always steps at 60 Hz) |
| `--config FILE` | `panel-games.toml` if present | per-game settings (size, bezel), see below |

Without `--ddp` only the screen output runs.

### With colorlight-driver

In the driver's `ledpanels.toml` enable the DDP listener (`ddp_port = 4048`) and restart it, then:

```
uv run panel-games serve --ddp <pi-ip> --target 768x64
```

`--target` must match the driver's canvas (768x64 for two stacked 768x32 strips). The game canvas is mapped onto
the target by the sender: the largest whole-number nearest-neighbour upscale that fits, centered (scale 1 when
the sizes match). A game larger than the target is center-cropped, never shrunk, but in practice it cannot be
started: the catalog marks it unavailable for that target. On 768x64 that leaves Panel Flap Resurrection and
Panel Hop (which has a 768x64 layout); Austin's other games need 768x128.

## Bezel

The screen output draws a bezel: the graphic around and between the game's picture. It is generated (a vertical
gradient plus a repeating pattern), so it fits any size:

- **Strip gaps:** `strip_height` cuts the picture into strips of that many rows (the physical LED strip height)
  with `gap * strip_height` rows of bezel between them, so the TV looks like the real strips. Panel Flap Resurrection uses
  32-row strips, a 1/5 gap (6 rows) and full width on a dark gradient, so a 768x64 game shows as 768x70.
- **Letterboxing:** `aspect = "16:9"` pads the screen frame with bezel to that ratio instead of black bars.
- **On the panels:** strip gaps are never sent by DDP (the real strips have their own). With `ddp = true` the
  bezel also fills the unused area when a game is smaller than `--target`; by default that area is black.

| setting | default | |
|---|---|---|
| `strip_height` | `0` (one piece) | rows per strip |
| `gap` | `0.2` | gap between strips, as a fraction of `strip_height` |
| `aspect` | none | screen ratio to letterbox to, e.g. `"16:9"` |
| `ddp` | `false` | fill the DDP letterbox with bezel too |
| `colors` | `["#202028", "#08080c"]` | gradient top and bottom |
| `pattern` | `diagonal` | `diagonal`, `checker`, `dots` or `none` |
| `pattern_size` | `8` | pixels per pattern cell |
| `pattern_light` | `10` | how much brighter (0-255) the pattern cells are |

A game sets its own defaults with a `bezel = {...}` class attribute.

## Per-game config

`--config FILE` (or `panel-games.toml` in the working directory) is TOML with one table per game id plus
`default` for all games. Precedence: built-in defaults, then `[default.bezel]`, then the game's own `bezel`,
then `[<id>.bezel]`. `size` restricts which canvas size a game is built at. Unknown games or settings stop the
server with an error. See `panel-games.example.toml`:

```toml
[default.bezel]
aspect = "16:9"            # letterbox every game on the TV

[resurrection]
size = [768, 64]           # always the two-strip version

[resurrection.bezel]
colors = ["#141022", "#000000"]
pattern = "dots"
```

## Games

Most games use a D-pad with A and B; others replace it with something better suited (answer buttons, a drawing
canvas, a tap button, a touch pad). Each phone keeps one player slot, and a game with no input for 2 minutes
ends and blanks the outputs. Multiplayer games wait in a lobby until the **host presses START** on their phone.
The host is the first player to join (if they leave, the next in line). Anyone joining mid-round plays the next one.

- **Panel Flap Resurrection** (768x32, 768x64 or 768x128): single player. The screen is static: the bird flies left to
  right, each level is one screen of pipes, 3 levels per stage, 16 themed stages (moving gaps, coins, darkness,
  flipped gravity, wind, low gravity) then endless. 3 lives, coins give +5 and every 20 an extra life. On a
  64-pixel-tall (or taller) canvas the bottom half is the underworld: drop down and you become skeleton bird,
  dodging magma pillars and hanging fire whose shared hole moves in sync (+2 per column passed down there). A or
  Up flaps, START pauses, Left/Right pick the starting stage (among the ones unlocked) before the first flap.
- **Panel Run**: an original one-screen platformer (96x16 tiles of 8 pixels, the whole level always visible).
  Up to 4 players race to the flag. Run, jump, stomp the walkers, hit coin boxes.
- **Pong**: 2 players; the computer plays any empty side.
- **Panel Flap**: up to 4 birds share one course of pipes. Tap A to flap. When a bird reaches the right edge play
  pauses and the whole screen scrolls over, so the course crosses every panel and keeps getting harder. The last
  bird flying wins.
- **Panel Chomp**: a random maze across the whole display (a new one each round), up to 4 players, one life each.
  Eat the dots, avoid the wisps, grab a corner power pellet to chase them for a few seconds. If the maze is
  cleared the game ends on a podium (1st, 2nd, 3rd by score, survivors ahead of anyone who was caught); if every
  player is caught, the last one standing wins.
- **Panel Bomber**: Bomberman-style arena across the whole display, up to 4 players. Drop bombs (A), blow up
  crates for power-ups (bigger blasts, more bombs, speed), last player standing wins. After 75 seconds the arena
  closes in from the edges so rounds always end.
- **Panel Sweep**: Minesweeper for up to 6 players on one shared board (48x7 squares). A reveals (or clears around
  a satisfied number), B flags. Revealing scores points, each mine costs one of two lives and stuns you for 3
  seconds, correct flags earn +10 and wrong ones cost 10 at the end. Clear the board for a podium.
- **Panel Trivia**: up to 8 players answer on their phones (four colored buttons). Faster correct answers score
  more (500 to 1000). The host picks the deck (mixed, nerd, engineering, movies and TV, sci-fi, fantasy) and the
  number of questions with the arrows, then presses START. Ends on a podium.
- **Panel Hop**: Frogger across the whole display, up to 8 players. Cross a chain of biomes (meadow, river, forest,
  swamp, desert, snow, volcano, then a huge multilane city highway) to the finish pond. Difficulty escalates toward
  the end, there are checkpoints and 3 lives, and cars flash a warning before entering the screen. The test suite
  includes a bot that proves every stretch is crossable (`LED_SLOW_TESTS=1` runs extra seeds).
  On 768x64 (two stacked strips) it uses the same lanes and biomes with 7 rows instead of 14 and an 8-pixel HUD;
  with half the room to dodge, city traffic is 65% as dense and water moves at half speed, which the same bot
  crosses on every seed tried (24 of 24).
- **Panel Sketch**: Pictionary for up to 8 players. The server picks the drawer, who draws a secret word on their
  phone while everyone else types guesses. Every player draws once per cycle (host picks 1 to 3 cycles with the
  arrows). Faster correct guesses score more, the drawer earns a bonus, and letters are revealed as time runs out.
  Guessers never receive the word.
- **Panel Tug of War**: 2 to 8 players are split into two teams and tap a big button on their phones to drag a
  flag along a rope. A team's pull is its taps divided by its size, so uneven teams are fair. Drag the flag to the
  far line to win a round at once, otherwise the side holding the flag when the timer ends wins (a dead heat goes
  to overtime). The host picks best of 1, 3 or 5 (LEFT/RIGHT) and 10 to 30 seconds a round (UP/DOWN).
- **Panel Hockey**: air hockey across the whole display for up to 4 players. Players 1 and 3 defend the left goal,
  2 and 4 the right, and the computer plays any side with nobody on it. Each phone shows a touch pad: slide your
  finger to move your mallet. The host picks first to 3, 5, 7 or 10 goals with the arrows.

All of Austin's games declare `size = (768, 128)`.

Trivia questions come from Open Trivia DB (https://opentdb.com, CC BY-SA 4.0), downloaded once into
`src/panel_games/data/trivia.json` so the game works offline. Refresh or enlarge it with
`python scripts/fetch_trivia.py`; `--retag` re-sorts the saved questions into decks without downloading.

## Adding a game

Create a class in `src/panel_games/games/` and list it in `REGISTRY` in `games/__init__.py`:

```python
class MyGame:
    id = "mygame"
    title = "My Game"
    description = "One line for the game list."
    max_players = 4
    controls = "Up / Down to move, A to fire"
    size = (768, 128)            # or a list in order of preference: [(768, 64), (768, 32)]
    bezel = {"strip_height": 32}  # optional screen bezel defaults (see Bezel)

    def __init__(self, width, height, seed=None): ...   # built at the chosen size
    def update(self, dt, pads): ...                     # fixed 1/60 s steps
    def render(self, canvas): ...                       # canvas: (height, width, 3) uint8 RGB numpy array
```

The host builds the game at the first declared size that fits inside `--target`; if none fits the game is
unavailable. `pads` has one entry per player slot: `None` if nobody joined, else the held buttons as a bitmask
(`LEFT 1, RIGHT 2, UP 4, DOWN 8, A 16, B 32, START 64, X 128, Y 256`; see `games/pad.py`). `games/base.py` has
the shared host-START `Lobby`, tile movement, podium and text helpers; `games/sprites.py` builds sprites from text.

Optional hooks:

- `info()` returns public state (lobby, scores, timers), merged into `/api/status` and `/api/game/state`.
- `private_info(slot)` returns what only that player may see (the word to draw, their own team).
- `on_message(slot, data)` receives free-form messages from `POST /api/game/message` (taps, drawing strokes,
  guesses, touch positions). Validate everything: any phone can send anything. Keep per-player rate limits in
  the game (see `tug.py` and `sketch.py`).
- `layout` picks the phone controller: `dpad` (default), `quiz` (four answer buttons), `sketch` (draw and guess),
  `tug` (tap button) or `air` (touch pad). A new layout needs a pane in `static/index.html`; the page polls
  `/api/game/state` to draw it.

Tests follow `tests/test_games_multiplayer.py` (`begin`, `pads_for`, `DT`). Run them with `uv run pytest`.

## HTTP API

POST requests need the header `X-Requested-With: panel-games` and a JSON body.

| endpoint | body | |
|---|---|---|
| GET `/` | | phone controller page |
| GET `/screen` | | screen page (polls the frame, pixel-scaled to the window) |
| GET `/api/games` | | catalog: id, title, description, controls, layout, `sizes`, `available`, `size` for this target |
| GET `/api/status` | | `{game, error, target, urls}`; `game` is null when nothing runs |
| GET `/api/screen.png?since=N` | | latest frame as PNG (`X-Frame` header = frame number; 204 if still N) |
| POST `/api/game/start` | `{id}` | start a game (replaces the running one); 400 if unknown or unavailable |
| POST `/api/game/stop` | `{}` | stop and blank |
| POST `/api/game/join` | `{}` | take a player slot: `{slot, token, game}` |
| POST `/api/game/input` | `{token, keys}` | held buttons (0-511); send every ~300 ms as a heartbeat |
| POST `/api/game/message` | `{token, data}` | free-form object for the game, at most 40 kB |
| POST `/api/game/state` | `{token}` | public state plus this player's private info |
| POST `/api/game/leave` | `{token}` | free the slot |

Limits (per client IP): game endpoints 60 requests/s (burst 90), start/stop one every 2 s (burst 3). Player
slots expire after 20 s without a request, held buttons are released after 1 s without an update, and a phone
(by IP) that joins again gets its existing slot back; requests from this computer can take several slots, which
is handy for testing in several tabs.

**Access is LAN-open**: anyone who can reach the port can start, stop and play games. There are no admin or
guest roles (Austin's PIN login and guest windows were dropped). Use `--bind 127.0.0.1` or a firewall if that
matters on your network.

## Layout

| file | |
|---|---|
| `cli.py` | `panel-games serve` |
| `host.py` | runs one game at a time on a fixed-rate loop thread, idle timeout, pushes frames to the outputs |
| `bezel.py` | generated bezel: strip gaps, letterboxing, gradient and pattern |
| `outputs.py` | `DDPOutput` (maps the game canvas onto the target and sends it), `ScreenOutput` (latest frame for `/screen`) |
| `ddp.py` | ~30-line DDP sender (1440-byte packets, PUSH on the last) |
| `web.py` | HTTP server, rate limiter |
| `games/` | the games, `pad.py` (player slots and controller state), `base.py`, `sprites.py` |
| `static/` | `index.html` (phone controller), `screen.html` |

## License

GNU General Public License v3.0, see `LICENSE`. The games are derived from Austin's
https://github.com/Austinrocket/Colorlight-LED-Panel-Drivers (GPLv3). The bundled trivia questions are from Open
Trivia DB under CC BY-SA 4.0.
