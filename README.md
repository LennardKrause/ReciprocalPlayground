# ReciprocalPlayground

## Very early version of a reciprocal space playground to:
 - explore diffraction geometry
 - understand the reciprocal space
 - teach single crystal X-ray diffraction

>[!NOTE]
>Check the [releases page](https://github.com/LennardKrause/ReciprocalPlayground/releases), download the latest *.whl and install it using pip.<br>
>```python3 -m pip install reciprocalplayground-1.0.0-py3-none-any.whl```

## Version 1.0.0 (28.10.2026)
 - First release candidate
## Version 0.1.1 (24.10.2026)
 - Change reciprocal lattice point size and alpha channel from gui.
 - Option to fix orientation matrix.
## Version 0.1.0 (22.10.2026)
 - Mainly fixing bugs and leaving the program in a better state as I've hopefully introduced less new bugs than were before!
 - You can right-click a spot currently in diffraction condition to set the center of rotation.
 - Hold shift to slow down rotation.
## Version 0.0.8 (04.12.2025)
 - Fixed the 2-Theta rotating some labels.
 - Changed the behaviour of the hkl labels and allowed them to be toggled.
 - Added labels to the detector.
 - Added a polarisation factor.
 - Added a simple data collection strategy table.
 - Added a crystal at the ewald center (it's there, you can't get rid of it yet!).
 - Added a placeholder detector screen.
 - FIXED: Wrong resolution cut-off applied to hkl generation.
 - Added more SFX options (parameterised partiality correction, of some kind!).
 - Calculation of symmetry equivalent reflections (and completeness) unified and fixed.
 - Detector parameters adjustable via GUI.
 - Hotkeys! Use F1-F2-F3 to look down the cell axes a-b-c, F4-F5-F6 to look down on Omega, Chi and Phi.

## How it looks
![Example image](https://github.com/LennardKrause/ReciprocalPlayground/blob/main/example.png)

>[!WARNING]
>This comes with plenty of bugs!

I hope you like it anyway.
