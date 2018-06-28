# Graphatoms
### a tool for manipulating MultiGraph representations of Atomic structures

## Motivation
MultiGraphs are a generalization of Graphs (the formal structure with Nodes and
  Edges) by allowing multiple edges to exist between any given two Nodes. Graph
  representations of chemical structures are prevalent in Cheminformatics because
  they allow us to talk about molecular structure at a high level (one can
  identify two water molecules as both being the same thing, i.e. water molecules,
  by showing they have an identical graph ... even if one has slightly different
  coordinates).

Graphs are sufficient to describe molecules, but of bulk and surface
chemical structures require a more expressive representation. This is due to the
fact that bulk and surface structures are infinitely sized - we represent them
in a periodic cell and need to account for the fact that atoms can bond to
atoms in these periodic images. With each atom in the unit cell as a representative
of the set of atoms in all periodic images, we can describe the structure as a
Multigraph, where we allow multiple bonds to exist between these representatives.

There are many potential benefits in surface science that could arise from
being able to talk about chemical structures at this high level.

## This repo
This repo is simply a collection of scripts that I have found useful in 
  generating and manipulating these multigraph representations. Certainly a work
  in progress with nothing guaranteed to be correct.

## Setup
Initialize your system with the following commands:

```
cd graphatoms
python3 -m venv .env
source .env/bin/activate
pip install -r requirements.txt
```
