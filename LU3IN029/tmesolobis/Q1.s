.data

str:		.asciiz	"dajdhaodoakd"
str_res:	.space	20



.text	

main:
	addiu	$29,$29,4
	lui	$8,0x1001
	addiu	$4, $8 ,0
	jal	len
#Affichage
	ori	$4,$2,0
	ori	$2,$0,1
	syscall
	
#sortie
	ori $2, $0, 10
	syscall
	
	
len:
#1 variable local + adresse de retour
	addiu	$29,  $29, -8
	sw	$31, 8($29)
	xor	$2, $2, $2
	

while:
# chargement du premier char
	addu	$9, $2, $4
	lb	$9, 0($9)
# condition
	beq	$9, $0, endwhile
# cpt += 1
	addiu	$2, $2, 1
# chargement du char suivant 

	j while
endwhile:
	lw	$31,8($29)
	addiu	$29, $29,8
	jr	$31
	
	
	
	
	