.data
chbis:  .space 20  # Espace alloué pour la recopie
N:      .word 2
ch:     .asciiz "Hello"

.text
    addiu   $29, $29, -4    # Variable locale i
    ori     $8, $0, 0       # $s8 (i) <- 0
    lui     $9, 0x1001      # Adresse de N
    lw      $9, 20($9)      # Lecture de N
    lui     $11, 0x1001     # @ch1
    ori     $11, $11, 24
    lui     $12, 0x1001     # @ch2

for:
    # Test i < N
    slt     $10, $8, $9      # Test i < N
    beq     $10, $0, finfor

    # ch2[i] = ch1[i]
    addu    $10, $11, $8     # @ch1[i]
    lb      $10, 0($10)      # Lecture ch1[i]
    addu    $13, $12, $8     # @ch2[i]
    sb      $10, 0($13)      # Recopie du caractère ch1[i] dans ch2[i]

    # i = i + 1
    addiu   $8, $8, 1       # i + 1
    j       for

finfor:
    addu    $13, $12, $8     # @ch2[i]
    sb      $0, 0($13)      # Caractère de fin de chaîne !

    # printf("%s", ch2)
    or      $4, $0, $12
    ori     $2, $0, 4
    syscall                 # Affichage de ch2

    addiu   $29, $29, 4     # Désallocation
    ori     $2, $0, 10
    syscall                 # Exit
