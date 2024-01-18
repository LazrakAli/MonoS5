.data

p:	.space 4
q:	.space 4

.text
#lecteur de p au clavier
	ori	$2,$0,5
	syscall
	lui	$8,0x1001
	sw	$2,0($8)
	
#lecture de q au clavier 
	ori	$2,$0,5
	syscall
	lui 	$8,0x1001
	sw	$2,4($8)
	
#chargement de p et q
	lw	$9,0($8)
	lw	$10,4($8)
	
#p < q
	sub	$11,$9,$10
	bgtz	$11,else
	#compteur=0
	xor	$11,$11,$11


loop:
	addi	$9,$9,1
	beq	$9,$10,then
	addu	$11,$11,$9
	j loop
then: 
#affichage compteur
	ori	$4,$11,0
	ori	$2,$0,1
	syscall
#exit
	ori	$2,$0,10
	syscall
else:
	ori	$4,$0,0
	ori	$2,$0,1
	syscall
#exit()
	ori	$2,$0,10
	syscall