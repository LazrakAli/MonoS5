.data

tab:	.word	2,4,2,10,24,220,1700,0

.text
	lui	$8,0x1001 #adresse tab
	xor	$9,$9,$9 #max=0
while:
	lw	$10,0($8)	
	beq	$10, $0, endwhile
	
	slt	$11,$9,$10
	beq	$11,$0,suite
	
	ori	$9,$10,0
suite:
	addiu	$8,$8,4
	j	while

endwhile:
#affichage du max
	ori 	$4,$9,0
	ori	$2,$0,1
	syscall
#exit()
	ori	$2,$0,10
	syscall