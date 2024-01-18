.data

val:	.word	7
tab:	.word	10,183,2,4,5,0,100,11,1,6,-1

.text

	lui	$8,0x1001
# val
	lw	$9,0($8)
		
# compteur
	xor	$15,$15,$15

# $14=-1
	ori	$14,$0,0
	addi	$14,$14,-1
while:
# chargement de tab[i]
	lw	$10,4($8)
	
# condition tab[i]!=-1
	beq	$10,$14,endwhile

# $11=val-tab[i]
	sub	$11,$9,$10

# incrementation i
	addiu	$8,$8,4
	
# if $11>0 incrementation du compteur
	blez	$11,while
	addi	$15,$15,1
	
# boucle
	j while

endwhile:
# affichage
	ori	$4,$15,0
	ori	$2,$0,1
	syscall
	
# exit()
	ori	$2,$0,10
	syscall