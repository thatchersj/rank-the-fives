library(data.table)

res <- readLines('TournamentResults.txt')
tres <- local({
  ts <- grep('^20..', res)
  lapply(setNames(seq_along(ts), res[ts]), function(i) {
    res[(ts[i] + 1):(c(ts - 1, length(res))[i + 1])]
  })
})
dtres <- rbindlist(lapply(tres, function(z) {
  if (z[1] == 'Not held') return(data.table(Round = character(), Name = character()))
  z <- gsub('\\([1-8]\\)', '', gsub('(\\w),(\\w)', '\\1, \\2', gsub(':', '', trimws(z[!z == '']))))
  rbind(
    if (any(grepl('Non-qualifiers', z, ignore.case = TRUE)))data.table(Round = 'NQ', Name = unlist(strsplit(strsplit(z[grep('Non-qualifiers', z, ignore.case = TRUE) + 1], ', ')[[1]], ' & '))),
    if (any(grepl('Last 16', z))) data.table(Round = 'L16', Name = unlist(strsplit(gsub('.* (bt|beat) ([A-z.\' -]* & [A-z.\' -]*) .*', '\\2', z[(grep('Last 16', z, ignore.case = TRUE) + 1):(grep('Quarter[- ]Final', z, ignore.case = TRUE) - 1)]), ' & '))),
    data.table(Round = 'QF', Name = unlist(strsplit(gsub('.* (bt|beat) ([A-z.\' -]* & [A-z.\' -]*) .*', '\\2', z[(grep('Quarter[- ]Finals', z, ignore.case = TRUE) + 1):(grep('Semi-Finals', z, ignore.case = TRUE) - 1)]), ' & '))),
    data.table(Round = 'SF', Name = unlist(strsplit(gsub('.* (bt|beat) ([A-z.\' -]* & [A-z.\' -]*) .*', '\\2', z[(grep('Semi-Finals', z, ignore.case = TRUE) + 1):(grep('Final$', z, ignore.case = TRUE) - 1)]), ' & '))),
    data.table(Round = 'F', Name = unlist(strsplit(gsub('.* (bt|beat) ([A-z.\' -]* & [A-z.\' -]*) .*', '\\2', z[(grep('Final$', z, ignore.case = TRUE) + 1)]), ' & '))),
    data.table(Round = 'W', Name = unlist(strsplit(gsub('([A-z.\' -]* & [A-z.\' -]*) (bt|beat) ([A-z.\' -]* & [A-z.\' -]*) .*', '\\1', z[(grep('Final$', z, ignore.case = TRUE) + 1)]), ' & ')))
  )
}), idcol = 'Tournament')[, Name := trimws(Name)]

dtres[Name %like% 'injury', Name := gsub('(.*) (quali|reach).*', '\\1', Name)]

dtres[, Name := trimws(gsub('\\(.*', '', Name))]

dtres[nchar(Name) > 21]
sort(unique(dtres$Name))
dtres
dtres[, Name2 := gsub(' ', '.', gsub('DE ', 'DE', gsub('SOUZA GIRAO', 'SOUZAGIRAO', gsub('VAN ', 'VAN', toupper(Name)))))]
dtres[, Initial := ifelse(grepl('\\.', Name2), sub('([^.]*)\\..*', '\\1', Name2), NA_character_)]
dtres[, Surname := ifelse(grepl('\\.', Name2), sub('^.*\\.([^.]*)$', '\\1', Name2), Name2)]
dtres[, (c('Year', 'Comp')) := tstrsplit(Tournament, ' ', type.convert = TRUE)]

dtres1 <- dtres[Year > 2012]
dtres1[, (c('Year', 'Comp')) := tstrsplit(Tournament, ' ', type.convert = TRUE)]
dtres1[, Comp := factor(Comp, levels = c('Northern', 'Kinnaird', 'London'), ordered = TRUE)]
dtres1[, LastHeld := max(Year), by = Comp]
#fwrite(dtres1, 'dtres2.csv')

CJ2 <- function(x,y,unique=FALSE) {
  if (unique) {
    x <- unique(x)
    y <- unique(y)
  }
  out = CJ(x = seq(nrow(x)), y = seq(nrow(y)))
  return(cbind(x[out$x], y[out$y]))
}

dtres2 <- dtres1[
  CJ2(dtres1[, .(Tournament)], dtres1[, .(Initial, Surname)], unique = TRUE),
  on = .(Tournament, Initial, Surname)
]
dtres2[, .N, by = .(Initial, Surname, Tournament)][order(N)]
dtres2 <- melt(dcast(dtres2, Initial + Surname ~ Tournament, value.var = 'Round', fill = 'DNS'), id.vars = 1:2, variable.name = 'Tournament', value.name = 'Round')
dtres2[, (c('Year', 'Comp')) := tstrsplit(Tournament, ' ', type.convert = TRUE)]
dtres2[, Comp := factor(Comp, levels = c('Northern', 'Kinnaird', 'London'), ordered = TRUE)]
dtres2[, LastHeld := max(Year), by = Comp]
dtres2[is.na(Round), Round := 'DNS']
dtres2[(LastHeld - Year) > 7 & Round == 'DNS', Round := 'NA']
dtres2[order(Year, Comp), Round := ifelse(1:.N < which.min(Round %in% c('DNS', 'NA') | (LastHeld - Year) > 7), ifelse(Round == 'DNS', 'NA', Round), Round), by = .(Initial, Surname)]

K1 <- c(W = 15, F = 13.5, SF = 9.6, QF = 5.6, L16 = 2, NQ = 0, DNS = 0, "NA" = 0)
K2 <- c(W = 15, F = 15, SF = 12, QF = 8, L16 = 4, NQ = 2, DNS = 1, "NA" = 0)

L1 <- c(W = 10, F = 8, SF = 5.2, QF = 3, NQ = 0, DNS = 0, "NA" = 0)
L2 <- c(W = 10, F = 10, SF = 8, QF = 6, NQ = 2, DNS = 1, "NA" = 0)

N1 <- c(W = 10, F = 8, SF = 5.2, QF = 3, NQ = 0, DNS = 0, "NA" = 0)
N2 <- c(W = 10, F = 10, SF = 8, QF = 6, NQ = 2, DNS = 1, "NA" = 0)

decay_factor <- c(1, 0.85, 0.7, 0.55, 0.4, 0.25, 0.1, rep(0, 993))

rpa <- function(Comp, Round) {
  mapply(
    FUN = function(C,R) list(
      Kinnaird = K1,
      London = L1,
      Northern = N1
    )[[as.character(C)]][R],
    Comp,
    Round
  )
}
poss <- function(Comp, Round) {
  mapply(
    FUN = function(C,R) {
      list(
        Kinnaird = K2,
        London = L2,
        Northern = N2
      )[[as.character(C)]][R]
    },
    Comp,
    Round
  )
}

rank_ <- function(dt, decay_factor) {
  RANKINGS <- dt[,
    .(
      RPA = sum(decay_factor[1 + LastHeld - Year] * rpa(Comp, Round)),
      POSS = sum(decay_factor[1 + LastHeld - Year] * poss(Comp, Round)),
      CompsPlayed = sum(!Round[(LastHeld - Year) < 7] %in% c('DNS', 'NA')),
      ConsecutiveCompsMissed = .SD[order(-Year, -Comp), sum(rleid(Round) == 1) * (Round[1] == 'DNS')],
      CompsMissed_Last6 = .SD[order(-Year, -Comp)][1:6, sum(Round == 'DNS')],
      CompsMissed_Last9 = .SD[order(-Year, -Comp)][1:9, sum(Round == 'DNS')]
    ),
    by = .(Initial, Surname)
  ][, PC := (RPA / POSS) * 100][]

  RANKINGS[, PC2 := PC / 1000^(ConsecutiveCompsMissed >= 3 | CompsPlayed < 3)]

  RANKINGS[, RANK := rank(-PC, ties.method = 'min')]
  RANKINGS[, RANK2 := rank(-PC2, ties.method = 'min')]
  RANKINGS[, RANK3 := frank(.SD, -PC2, -POSS, ties.method = 'first')]

  RANKINGS[order(-PC2)]
}

dtres2[dtres2[, .(YC = factor(paste(Year, Comp), levels = paste(Year, Comp), ordered = TRUE)), keyby = .(Year, Comp)], YC := i.YC, on = .(Year, Comp)]

fwrite(
  setnames(
    dcast(
      copy(dtres2)[Round == 'NA', Round := ''][Round == 'DNS', Round := 'P'],
      Initial + Surname ~ YC,
      value.var = 'Round'
    ),
    levels(dtres2$YC),
    substr(levels(dtres2$YC), 3, 6)
  )[
    rank_(dtres2, decay_factor)[
      order(RANK3),
      .(
        Initial,
        Surname,
        RPA,
        POSS,
        Played = CompsPlayed,
        MissedLast = ConsecutiveCompsMissed,
        PC,
        PC2,
        RANK,
        RANK2,
        RANK3
      )
    ],
    on = .(Initial, Surname)
  ],
  sprintf('SJT_FivesRankings_%s.csv', format(Sys.Date(), format = '%Y%m%d'))
)
