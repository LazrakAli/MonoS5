.data
a: 	.space	4
b:	.space	4

.text
#lecteur de a au clavier
	ori	$2,$0,5
	syscall
	lui	$8,0x1001
	sw	$2,0($8)
	
#lecture de b au clavier 
	ori	$2,$0,5
	syscall
	lui 	$8,0x1001
	sw	$2,4($8)
	lw	$9,0($8)
	lw	$10,4($8)
	ori	$12,$0,1
	
while:
	beq	$9,$10, endwhile
	sub	$11,$9,$10
	bltz	$11,else
	ori	$9,$11,0
	j while
	
else:
	sub	$11,$10,$9
	ori	$10,$11,0
	j while
	
endwhile:	
#affichage
	ori 	$4,$11,0
	ori	$2,$0,1
	syscall
#exit
	ori	$2,$0,10
	syscall	