# TODO

[ ] It fails as inference continues to infty, because logits will collapse to -infty / +infty once we get really good onces, and the computer just won't have it. If we use probability, then we're actually more numerically stable. That's a new one.
