.data

n:	.word 0xFEDCBA98

.text

	lui	$8,0x1001
	lw	$9,0($8) #n
	xor	$10,$10,$10 #compteur a 0
	
count:
	#test si le bit de point faible est egale a 1
	andi	$11,$9,1 # $9 & 0b0001 -> 0b0001 si bit_point_faible==1 sinon 0
	beq	$11,$0,saut
# on incremente le compteur
	addi	$10,$10,1
	
saut:	
	srl	$9,$9,1 #decalage de 1
	bne	$9,$0,count #verifie la fin de boucle

# affichage	
	ori	$4,$10,0
	ori	$2,$0,1
	syscall
# exit
	ori	$2,$0,10
	syscall
	
	
	