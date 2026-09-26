# Mixed-sampler ballots (8 per question) vs 8 card ballots -- split-half on the frozen set

arms with full coverage: ['card', 'T1.2k20', 'T1.2p.7', 'T1.2p.8', 'T1.1k20', 'T1.6p0.2', 'T1.6p0.3', 'T1.6p0.4', 'T1.6p0.5', 'T1.6p0.6', 'T1.6p0.7'] | half A n=703 | half B n=704 | holdout n=322 | candidates 185

top 8 on half A (selection half):
  cardx2+T1.2k20x2+T1.2p.8x2+T1.6p0.2x2    +1.56 pp
  cardx2+T1.2k20x2+T1.2p.8x2+T1.6p0.4x2    +1.56 pp
  cardx2+T1.2k20x2+T1.2p.7x2+T1.2p.8x2     +1.42 pp
  cardx2+T1.2k20x2+T1.2p.8x2+T1.6p0.5x2    +1.42 pp
  cardx2+T1.2p.8x2+T1.1k20x2+T1.6p0.6x2    +1.42 pp
  cardx2+T1.2p.8x2+T1.1k20x2+T1.6p0.5x2    +1.14 pp
  cardx2+T1.2p.8x2+T1.6p0.2x2+T1.6p0.6x2   +1.14 pp
  cardx2+T1.2k20x2+T1.6p0.2x2+T1.6p0.5x2   +1.00 pp

SELECTED on A: cardx2+T1.2k20x2+T1.2p.8x2+T1.6p0.2x2
CONFIRMATION on B: mix maj@8 0.7259 vs card 0.7202  delta +0.57 pp  SE 0.78  t=0.73  p=0.4656  (n=704)
A-PRIORI mix card4+T1.2k20x4 on B: 0.7230 vs 0.7202  delta +0.28 pp  p=0.7153
A-PRIORI mix on full frozen set: 0.7306 vs 0.7292  delta +0.14 pp  p=0.7774  (n=1407)
HOLDOUT: not yet generated for the selected mix's arms
