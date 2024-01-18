.data

ch:	.byte	0x31,0x32,0x33,0x34,0x00

.text
	addiu	$29,$29,-9

# on enregistre dans r9 et r10 la valeur 0 pour initialiser i val
	xor	$9,$9,$9 
	xor	$10,$10,$10

# initialisation des valeurs

	sw	$9,0($29) # i = 0
	sw	$10,4($29) # val = 0
	
# chargment de tab

	lui	$8, 0x1001
	ori	$15,$0,10 # pour multiplier par 10 on le met hors boucle 

while:
	lbu	$11,0($8) # tab[0]
	
# ch[i] != 0
	beq	$11, $0, endwhile
	
	ori	$12,$11,0 # c = ch[i];
	andi	$12,$12,0x000F # c = c & 0x0F;
	sb	$12,8($29) # on enregistre c en memoire
	
# val = val * 10 + c;
	multu	$10,$15
	mflo	$10
	addu	$10,$10,$12
	sb	$10,4($29)
	addiu	$9,$9,1
	sb	$9,0($29)

# passage au char suivant et boucle
	addiu	$8,$8,1
	j	while
	
endwhile:
	ori 	$4,$10,0
	ori	$2,$0,1
	syscall
	
# on libere la memeoire
	addiu	$29,$29,9
# exit	
	ori	$2,$0,10
	syscall

	
	
	
	
	
	