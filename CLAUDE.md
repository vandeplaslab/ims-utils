# ims-utils
Provides numerical routines for mass spectra: centroiding, baseline, smoothing, resampling, lock-mass correction, peak filtering and batch normalization.

## Role in ecosystem
Foundation processing utility used by readers and pre-processing libraries.
Internal deps: koyo (imported, undeclared: imzy)
Depended on by: imspy-pre, imzy (imported, undeclared: annotine, image2image, imspy-core, imspy-data, imspy-misc, imspy-multi, imspy-pasef, imspy-stats, imspy-train, ionglow, pyspa, qucee)

## Key abstractions
- `spectrum` — centroiding (`oms_centroid`, `parabolic_centroid`, `maxima_centroid`), noise estimation, `resample_ppm`, `interpolate_ppm`, `downsample`
- `filters` — `FilterBase` and intensity/smoothing filters (`SavitskyGolayFilter`, `GaussianSmoothFilter`, ...)
- `lockmass` (`LockmassEstimator` variants), `robust_lockmass.RobustLockmassEstimator`
- `batchnorm` (`ComBat`, `ReComBat`), `internorm.CentroidMeanFoldInterNorm`
- `peak_filter.PeakFilter`, `baseline`, `smooth`, `deisotope`, `detect`, `kendrick`

## Do not re-explain
When working in this repo, assume familiarity with: imzy, koyo
