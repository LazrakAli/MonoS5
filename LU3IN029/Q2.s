.data

str:		.asciiz	"abcdef"
str_res:	.space	20

.text	

main:
	addiu	$29, $29, -16
	
	lui	$4, 0x1001
	jal	len
	ori	$6, $2, 0
	
	lui	$4, 0x1001
	ori	$5, $4, 0x0006
	addiu	$6, $6, -1
	jal	miroir
	
	lui	$4, 0x1001
	ori	$4, $4, 0x0006
	ori	$2, $0, 4
	syscall
	
	addiu	$29, $29, 16
	ori	$2, $0, 10
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
	
		
miroir:
	addiu	$29, $29, -8
	sw	$31, 4($29)
	xor	$7, $7, $7

while1:
	bltz	$6, endWhile1
	
	addu	$8, $4, $6
	addu	$9, $5, $7
	
	lbu	$8, 0($8)
	sb	$8, 0($9)
	addiu	$7, $7, 1
	addiu	$6, $6, -1
	j	while1
	
endWhile1:
	lw	$31, 4($29)
	addiu	$29, $29, 8
	jr	$31
